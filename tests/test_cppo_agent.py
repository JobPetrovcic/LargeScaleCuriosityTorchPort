
import pytest
import numpy as np
import torch
import gym
import os
from cnn_policy import CnnPolicy
from dynamics import Dynamics
from cppo_agent import PpoOptimizer, RewardForwardFilter
from auxiliary_tasks import FeatureExtractor
from utils import activ
import logger

class MockEnv:
    def __init__(self, nenvs, ob_space, ac_space):
        self.nenvs = nenvs
        self.num_envs = nenvs
        self.observation_space = ob_space
        self.action_space = ac_space
        self.steps = 0
        
    def reset(self):
        self.steps = 0
        shape = (self.nenvs,) + self.observation_space.shape
        return np.zeros(shape, dtype=np.uint8)
        
    def step_async(self, actions):
        pass
        
    def step_wait(self):
        self.steps += 1
        shape = (self.nenvs,) + self.observation_space.shape
        obs = np.random.randint(0, 255, shape, dtype=np.uint8)
        rews = np.ones(self.nenvs, dtype=np.float32)
        dones = np.zeros(self.nenvs, dtype=bool)
        infos = [{'episode': {'r': 10, 'l': 10}} for _ in range(self.nenvs)]
        return obs, rews, dones, infos
    
    def close(self):
        pass

class MockAuxiliaryTask(torch.nn.Module):
    def __init__(self, policy):
        super().__init__() # type:ignore
        self.policy = policy
        self.hidsize = policy.hidsize
        self.ob_space = policy.ob_space
        self.ac_space = policy.ac_space
        self.ob_mean = torch.as_tensor(policy.ob_mean).float()
        self.ob_std = torch.as_tensor(policy.ob_std).float()
        self.feat_dim = 32
        
    def get_features(self, x):
        if x.dim() == 5:
            return torch.randn(x.shape[0], x.shape[1], self.feat_dim)
        else:
            return torch.randn(x.shape[0], self.feat_dim)
            
    def get_loss(self, obs, last_obs, acs):
        return torch.tensor(0.1)

def test_reward_forward_filter():
    gamma = 0.99
    rff = RewardForwardFilter(gamma)
    
    rews1 = np.array([1.0, 2.0])
    out1 = rff.update(rews1)
    assert np.allclose(out1, rews1)
    
    rews2 = np.array([1.0, 1.0])
    out2 = rff.update(rews2)
    # out2 = out1 * gamma + rews2 = [1, 2] * 0.99 + [1, 1] = [0.99+1, 1.98+1] = [1.99, 2.98]
    expected = rews1 * gamma + rews2
    assert np.allclose(out2, expected)

def test_ppo_optimizer_init():
    nenvs = 2
    ob_space = gym.spaces.Box(0, 255, (84, 84, 5), dtype=np.uint8)
    ac_space = gym.spaces.Discrete(4)
    hidsize = 64
    feat_dim = 32
    ob_mean = np.zeros((5, 84, 84), dtype=np.float32)
    ob_std = np.ones((5, 84, 84), dtype=np.float32)
    
    policy = CnnPolicy(ob_space, ac_space, hidsize, ob_mean, ob_std, feat_dim, 
                       layernormalize=False, nl=activ)
    
    aux_task = MockAuxiliaryTask(policy)
    dynamics = Dynamics(aux_task, predict_from_pixels=False, feat_dim=feat_dim)
    
    ppo = PpoOptimizer(scope="ppo", ob_space=ob_space, ac_space=ac_space, stochpol=policy,
                       ent_coef=0.01, gamma=0.99, lam=0.95, nepochs=3, lr=1e-4, cliprange=0.2,
                       nminibatches=4, normrew=True, normadv=True, use_news=True,
                       ext_coeff=1.0, int_coeff=1.0, nsteps_per_seg=5, nsegs_per_env=1,
                       dynamics=dynamics)
    
    assert ppo.optimizer is not None
    assert len(ppo.params) > 0

def test_ppo_optimizer_start_interaction():
    nenvs = 2
    ob_space = gym.spaces.Box(0, 255, (84, 84, 5), dtype=np.uint8)
    ac_space = gym.spaces.Discrete(4)
    hidsize = 64
    feat_dim = 32
    ob_mean = np.zeros((5, 84, 84), dtype=np.float32)
    ob_std = np.ones((5, 84, 84), dtype=np.float32)
    
    policy = CnnPolicy(ob_space, ac_space, hidsize, ob_mean, ob_std, feat_dim, 
                       layernormalize=False, nl=activ)
    
    aux_task = MockAuxiliaryTask(policy)
    dynamics = Dynamics(aux_task, predict_from_pixels=False, feat_dim=feat_dim)
    
    ppo = PpoOptimizer(scope="ppo", ob_space=ob_space, ac_space=ac_space, stochpol=policy,
                       ent_coef=0.01, gamma=0.99, lam=0.95, nepochs=3, lr=1e-4, cliprange=0.2,
                       nminibatches=4, normrew=True, normadv=True, use_news=True,
                       ext_coeff=1.0, int_coeff=1.0, nsteps_per_seg=5, nsegs_per_env=1,
                       dynamics=dynamics)
    
    env = MockEnv(nenvs, ob_space, ac_space)
    
    # Configure logger to avoid NoneType error in Recorder
    logger.configure("/tmp/test_cppo_agent")
    
    ppo.start_interaction(env, dynamics)
    
    assert ppo.rollout is not None
    assert ppo.rollout.nenvs == nenvs

def test_ppo_optimizer_step():
    n_channels = 3
    nenvs = 2
    ob_space = gym.spaces.Box(0, 255, (84, 84, n_channels), dtype=np.uint8)
    ac_space = gym.spaces.Discrete(4)
    hidsize = 64
    feat_dim = 32
    ob_mean = np.zeros((n_channels, 84, 84), dtype=np.float32)
    ob_std = np.ones((n_channels, 84, 84), dtype=np.float32)
    
    policy = CnnPolicy(ob_space, ac_space, hidsize, ob_mean, ob_std, feat_dim, 
                       layernormalize=False, nl=activ)
    
    aux_task = MockAuxiliaryTask(policy)
    dynamics = Dynamics(aux_task, predict_from_pixels=False, feat_dim=feat_dim)
    
    # nminibatches=1 to ensure batch size is valid for small rollout
    ppo = PpoOptimizer(scope="ppo", ob_space=ob_space, ac_space=ac_space, stochpol=policy,
                       ent_coef=0.01, gamma=0.99, lam=0.95, nepochs=1, lr=1e-4, cliprange=0.2,
                       nminibatches=1, normrew=True, normadv=True, use_news=True,
                       ext_coeff=1.0, int_coeff=1.0, nsteps_per_seg=5, nsegs_per_env=1,
                       dynamics=dynamics)
    
    env = MockEnv(nenvs, ob_space, ac_space)
    
    # Configure logger
    logger.configure("/tmp/test_cppo_agent_step")
    
    ppo.start_interaction(env, dynamics)
    
    # Mock rollout collection by filling buffers manually to avoid running env steps if we want speed,
    # but calling step() runs collect_rollout() which runs env steps.
    # MockEnv is fast enough.
    
    result = ppo.step()
    
    assert 'update' in result
    info = result['update']
    assert 'optimization_total_loss' in info
    assert 'optimization_policy_gradient_loss' in info
    assert 'optimization_dynamics_loss' in info
    assert info['n_updates'] == 1

