import functools
import os
import os.path as osp
import yaml
import torch
import numpy as np
import gym
import wandb

# Local imports
import logger
from monitor import Monitor
from common import set_global_seeds, NoopResetEnv

from auxiliary_tasks import FeatureExtractor, InverseDynamics, VAE, JustPixels
from cnn_policy import CnnPolicy
from cppo_agent import PpoOptimizer
from dynamics import Dynamics, UNet
from utils import random_agent_ob_mean_std, getsess
from wrappers import (
    MontezumaInfoWrapper, make_mario_env, make_robo_pong, make_robo_hockey,
    make_multi_pong, AddRandomStateToInfo, MaxAndSkipEnv, ProcessFrame84,
    ExtraTimeLimit, VecEnvAdapter
)

# Use AsyncVectorEnv from Gym
from gym.vector import AsyncVectorEnv

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def start_experiment(hps):
    # Setup logging and seeds
    logdir = logger.get_dir()
    if logdir is None:
        logdir = osp.join("/tmp", hps['exp_name'])
        logger.configure(dir=logdir)
    
    # Initialize wandb
    wandb.init(project="large-scale-curiosity", config=hps, dir=logdir)

    print(f"Results will be saved to {logdir}")
    
    # Set seeds
    seed = hps['seed']
    set_global_seeds(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create Trainer
    make_env = functools.partial(make_env_all_params, add_monitor=True, hps=hps)
    
    trainer = Trainer(make_env=make_env,
                      hps=hps,
                      num_timesteps=hps['num_timesteps'],
                      envs_per_process=hps['envs_per_process'])
    
    trainer.train()

class Trainer(object):
    def __init__(self, make_env, hps, num_timesteps, envs_per_process):
        self.make_env = make_env
        self.hps = hps
        self.envs_per_process = envs_per_process
        self.num_timesteps = num_timesteps
        self.device = getsess() # Returns torch device

        # 1. Setup Environment Variables (Spaces, Normalization stats)
        self._set_env_vars()

        # 2. Policy
        self.policy = CnnPolicy(
            scope='pol',
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            hidsize=512,
            feat_dim=512,
            ob_mean=self.ob_mean,
            ob_std=self.ob_std,
            layernormalize=False,
            nl=torch.nn.functional.leaky_relu
        ).to(self.device)

        # 3. Feature Extractor
        # Dictionary mapping string to class/partial
        feat_extractor_cls_map = {
            "none": FeatureExtractor,
            "idf": InverseDynamics,
            "vaesph": functools.partial(VAE, spherical_obs=True),
            "vaenonsph": functools.partial(VAE, spherical_obs=False),
            "pix2pix": JustPixels
        }
        
        FeatExtractorCls = feat_extractor_cls_map[hps['feat_learning']]
        
        self.feature_extractor = FeatExtractorCls(
            policy=self.policy,
            features_shared_with_policy=False,
            feat_dim=512,
            layernormalize=hps['layernorm']
        ).to(self.device)

        # 4. Dynamics
        # If pix2pix, use UNet, else standard Dynamics
        DynamicsCls = UNet if hps['feat_learning'] == 'pix2pix' else Dynamics
        
        self.dynamics = DynamicsCls(
            auxiliary_task=self.feature_extractor,
            predict_from_pixels=hps['dyn_from_pixels'],
            feat_dim=512
        ).to(self.device)

        # 5. Agent (Optimizer)
        self.agent = PpoOptimizer(
            scope='ppo',
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            stochpol=self.policy,
            use_news=hps['use_news'],
            gamma=hps['gamma'],
            lam=hps["lambda"],
            nepochs=hps['nepochs'],
            nminibatches=hps['nminibatches'],
            lr=hps['lr'],
            cliprange=0.1,
            nsteps_per_seg=hps['nsteps_per_seg'],
            nsegs_per_env=hps['nsegs_per_env'],
            ent_coef=hps['ent_coeff'],
            normrew=hps['norm_rew'],
            normadv=hps['norm_adv'],
            ext_coeff=hps['ext_coeff'],
            int_coeff=hps['int_coeff'],
            dynamics=self.dynamics
        )

    def _set_env_vars(self):
        # Create a dummy env to get spaces and statistics
        env = self.make_env(0, add_monitor=False)
        self.ob_space = env.observation_space
        self.ac_space = env.action_space
        
        print("Calculating random agent observation mean and std...")
        self.ob_mean, self.ob_std = random_agent_ob_mean_std(env)
        print("Done.")
        
        env.close()
        del env

    def train(self):
        # Create the actual Vector Environment
        # PpoOptimizer expects an object that acts like VecEnv
        
        env_fns = [functools.partial(self.make_env, i) for i in range(self.envs_per_process)]
        
        # Use Gym AsyncVectorEnv
        # We wrap it in VecEnvAdapter to ensure step() returns (obs, rew, done, info)
        # instead of (obs, rew, term, trunc, info)
        vector_env = AsyncVectorEnv(env_fns)
        vec_env_adapter = VecEnvAdapter(vector_env)
        
        self.agent.start_interaction(vec_env_adapter, dynamics=self.dynamics)
        
        while True:
            info = self.agent.step()
            
            # Logging
            if info['update']:
                logger.logkvs(info['update'])
                logger.dumpkvs()
            
            # Check termination
            total_steps = self.agent.n_updates * self.agent.rollout.nsteps * self.agent.rollout.nenvs
            if total_steps > self.num_timesteps:
                break

        self.agent.stop_interaction()


def make_env_all_params(rank, add_monitor, hps):
    # Factory function compatible with AsyncVectorEnv (takes no args usually, but partial handles it)
    # Note: rank is passed by the list comprehension in train()
    
    if hps["env_kind"] == 'atari':
        env = gym.make(hps['env'])
        # gym.make in modern versions usually includes NoFrameskip/etc logic if ID matches, 
        # but we follow the wrappers manual stack.
        
        # Original: assert 'NoFrameskip' in env.spec.id
        # In modern gym, env.spec.id might be None if created via some paths, but usually fine.
        
        # Note: Baselines wrappers (NoopResetEnv) are often specific.
        # We assume standard gym.Wrapper[Any, Any]s or imported wrappers handle most.
        # But 'wrappers.py' imported above should contain all necessary custom wrappers.
        
        # We rely on gym's default Atari preprocessing? No, code uses manual wrappers.
        # We assume 'env' is the raw environment.
        
        # In modern Gym, 'BreakoutNoFrameskip-v4' returns an environment that already outputs (210, 160, 3).
        
        env = NoopResetEnv(env, noop_max=hps['noop_max'])
        env = MaxAndSkipEnv(env, skip=4)
        env = ProcessFrame84(env, crop=False) # Config says crop=False in code logic? 
        # Original run.py: "env = ProcessFrame84(env, crop=False)"
        
        # from gym.Wrappers import FrameStack
        # env = FrameStack(env, 4)
        from wrappers import FrameStack
        env = FrameStack(env, 4)
        env = ExtraTimeLimit(env, hps['max_episode_steps'])
        
        if 'Montezuma' in hps['env']:
            env = MontezumaInfoWrapper(env)
        env = AddRandomStateToInfo(env)
        
    elif hps["env_kind"] == 'mario':
        env = make_mario_env()
    elif hps["env_kind"] == "retro_multi":
        env = make_multi_pong()
    elif hps["env_kind"] == 'robopong':
        if hps["env"] == "pong":
            env = make_robo_pong()
        elif hps["env"] == "hockey":
            env = make_robo_hockey()
    else:
        raise ValueError(f"Unknown env_kind: {hps['env_kind']}")

    if add_monitor:
        env = Monitor(env, osp.join(logger.get_dir(), '%.2i' % rank))
        
    return env

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load config
    hps = load_config(args.config)
    start_experiment(hps)