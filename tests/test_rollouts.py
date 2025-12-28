
import pytest
import numpy as np
import torch
import gym
from rollouts import Rollout
from cnn_policy import CnnPolicy
from dynamics import Dynamics
from auxiliary_tasks import FeatureExtractor

class MockEnv:
    def __init__(self, nenvs, ob_space, ac_space):
        self.nenvs = nenvs
        self.observation_space = ob_space
        self.action_space = ac_space
        self.steps = 0
        
    def reset(self):
        self.steps = 0
        # Return (N, H, W, C)
        shape = (self.nenvs,) + self.observation_space.shape
        return np.zeros(shape, dtype=np.uint8)
        
    def step_async(self, actions):
        self.actions = actions
        
    def step_wait(self):
        self.steps += 1
        shape = (self.nenvs,) + self.observation_space.shape
        obs = np.random.randint(0, 255, shape, dtype=np.uint8)
        rews = np.ones(self.nenvs, dtype=np.float32)
        dones = np.zeros(self.nenvs, dtype=bool)
        infos = [{'episode': {'r': 10, 'l': 10}} for _ in range(self.nenvs)]
        return obs, rews, dones, infos

class MockPolicy(torch.nn.Module):
    def __init__(self, ob_space, ac_space):
        super().__init__() # type:ignore
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.hidsize = 64
        # Register buffers to match CnnPolicy interface
        self.register_buffer('ob_mean', torch.zeros(ob_space.shape).permute(2, 0, 1)) # C, H, W
        self.register_buffer('ob_std', torch.ones(ob_space.shape).permute(2, 0, 1))
        
    def forward(self, x):
        # x: (B, T, C, H, W)
        B = x.shape[0]
        a_samp = torch.zeros(B, 1, dtype=torch.long)
        vpred = torch.zeros(B, 1, dtype=torch.float32)
        nlp = torch.zeros(B, 1, dtype=torch.float32)
        entropy = torch.zeros(B, 1, dtype=torch.float32)
        logits = torch.zeros(B, 1, self.ac_space.n)
        return a_samp, vpred, nlp, entropy, logits

class MockDynamics(torch.nn.Module):
    def __init__(self):
        super().__init__() # type:ignore
        
    def get_loss(self, obs, last_obs, acs):
        # obs: (B, T, C, H, W)
        # Returns (B, T)
        B, T = obs.shape[0], obs.shape[1]
        return torch.zeros(B, T, dtype=torch.float32)

def test_rollout_initialization():
    nenvs = 2
    nsteps_per_seg = 5
    nsegs_per_env = 1
    ob_space = gym.spaces.Box(0, 255, (84, 84, 1), dtype=np.uint8)
    ac_space = gym.spaces.Discrete(4)
    
    env = MockEnv(nenvs, ob_space, ac_space)
    policy = MockPolicy(ob_space, ac_space)
    dynamics = MockDynamics()
    
    rollout = Rollout(ob_space, ac_space, nenvs, nsteps_per_seg, nsegs_per_env, env, policy,
                      int_rew_coeff=1.0, ext_rew_coeff=1.0, record_rollouts=False, dynamics=dynamics)
    
    assert rollout.buf_obs.shape == (nenvs, nsteps_per_seg * nsegs_per_env, 1, 84, 84)
    assert rollout.buf_acs.shape == (nenvs, nsteps_per_seg * nsegs_per_env)

def test_rollout_collect():
    nenvs = 2
    nsteps_per_seg = 5
    nsegs_per_env = 1
    ob_space = gym.spaces.Box(0, 255, (84, 84, 1), dtype=np.uint8)
    ac_space = gym.spaces.Discrete(4)
    
    env = MockEnv(nenvs, ob_space, ac_space)
    policy = MockPolicy(ob_space, ac_space)
    dynamics = MockDynamics()
    
    rollout = Rollout(ob_space, ac_space, nenvs, nsteps_per_seg, nsegs_per_env, env, policy,
                      int_rew_coeff=1.0, ext_rew_coeff=1.0, record_rollouts=False, dynamics=dynamics)
    
    rollout.collect_rollout()
    
    # Check if buffers are filled
    # buf_obs should not be all zeros (except maybe first step if reset returns zeros)
    # But MockEnv returns random obs in step_wait.
    # Step 0: reset -> zeros.
    # Step 1..N: step_wait -> random.
    # buf_obs stores result of step t at index t.
    # Wait, rollout_step logic:
    # 1. env_get() -> obs (from prev step or reset)
    # 2. policy(obs) -> acs
    # 3. env_step(acs)
    # 4. buf_obs[:, t] = obs
    
    # So buf_obs[:, 0] is from reset (zeros).
    # buf_obs[:, 1] is from step 1 (random).
    
    assert (rollout.buf_obs[:, 0] == 0).all()
    # assert (rollout.buf_obs[:, 1] != 0).any() # Might fail if random returns 0, but unlikely for whole image
    
    # Check rewards
    # buf_ext_rews[:, t-1] = prevrews
    # t=0: no prevrews stored.
    # t=1: stores rews from step 0? No, env_get returns rews from step just taken.
    # In loop t=0: env_get (reset) -> rews=0.
    # t=1: env_get (step 1) -> rews=1. buf_ext_rews[:, 0] = 1.
    
    assert (rollout.buf_ext_rews[:, 0] == 1.0).all()
    
    # Check stats
    assert rollout.stats['epcount'] > 0

def test_rollout_calculate_reward():
    nenvs = 2
    nsteps_per_seg = 5
    nsegs_per_env = 1
    ob_space = gym.spaces.Box(0, 255, (84, 84, 1), dtype=np.uint8)
    ac_space = gym.spaces.Discrete(4)
    
    env = MockEnv(nenvs, ob_space, ac_space)
    policy = MockPolicy(ob_space, ac_space)
    dynamics = MockDynamics()
    
    rollout = Rollout(ob_space, ac_space, nenvs, nsteps_per_seg, nsegs_per_env, env, policy,
                      int_rew_coeff=0.5, ext_rew_coeff=2.0, record_rollouts=False, dynamics=dynamics)
    
    # Fill buffers manually
    rollout.buf_ext_rews.fill_(1.0)
    # MockDynamics returns 0 int reward.
    
    rollout.calculate_reward()
    
    # buf_rews = 2.0 * clamp(1.0) + 0.5 * 0 = 2.0
    assert (rollout.buf_rews == 2.0).all()

