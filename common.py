import random
import numpy as np
import gym
from typing import Any

def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)

class NoopResetEnv(gym.Wrapper[Any, Any]):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        # Check if action 0 is NOOP. 
        # Some envs might not have get_action_meanings, or 0 might not be NOOP.
        # We'll assume the user knows what they are doing if they use this wrapper.
        if hasattr(env.unwrapped, 'get_action_meanings'):
            if env.unwrapped.get_action_meanings()[0] != 'NOOP':
                print("Warning: NoopResetEnv: Action 0 is not NOOP.")

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        result = self.env.reset(**kwargs)
        
        # Handle (obs, info) vs obs
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            obs, info = result
            uses_new_api = True
        else:
            obs = result
            uses_new_api = False
            
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            # Use the env's random state if available, else numpy
            if hasattr(self.unwrapped, 'np_random'):
                noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
            else:
                noops = np.random.randint(1, self.noop_max + 1)
                
        assert noops > 0
        
        for _ in range(noops):
            step_result = self.env.step(self.noop_action)
            
            if len(step_result) == 5:
                step_obs, _, term, trunc, _ = step_result
                done = term or trunc
            elif len(step_result) == 4:
                step_obs, _, done, _ = step_result
            else:
                raise ValueError("Env returned unexpected number of values")
            
            if done:
                result = self.env.reset(**kwargs)
                if uses_new_api:
                    obs, info = result
                else:
                    obs = result
            else:
                obs = step_obs
                
        if uses_new_api:
            return obs, info
        return obs
