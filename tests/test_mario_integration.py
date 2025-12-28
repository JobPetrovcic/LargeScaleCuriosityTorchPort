import sys
import os
import pytest
import numpy as np
import gym
from typing import Any, Tuple, Dict

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers import MarioXReward

class MockMarioEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(240, 256, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(256)
        self.step_count = 0
    
    def reset(self, **kwargs):
        self.step_count = 0
        return np.zeros((240, 256, 3), dtype=np.uint8), {}
        
    def step(self, action):
        self.step_count += 1
        # Simulate some progress
        info = {
            "world": 1, 
            "stage": 1, 
            "x_pos": 40 + self.step_count
        }
        
        # Done on 2nd step
        done = self.step_count >= 2
        truncated = False
        
        return np.zeros((240, 256, 3), dtype=np.uint8), 0.0, done, truncated, info

def test_mario_episode_info_format():
    """
    Test that MarioXReward wrapper returns the correct format of episode information
    as expected by rollouts.py.
    
    rollouts.py expects:
    - info['retro_episode'] exists
    - info['retro_episode']['levels'] exists and is a set/list of tuples
    """
    
    # Setup
    base_env = MockMarioEnv()
    env = MarioXReward(base_env)
    
    # Reset
    env.reset()
    
    # Step 1: Not done
    _, _, done, _, info = env.step(0)
    assert not done
    assert "retro_episode" not in info
    
    # Step 2: Done
    _, _, done, _, info = env.step(0)
    assert done
    
    # Check keys
    assert "retro_episode" in info, "info should contain 'retro_episode' when done"
    assert "levels" in info["retro_episode"], "'retro_episode' should contain 'levels'"
    
    # Check types
    levels = info["retro_episode"]["levels"]
    assert isinstance(levels, (set, list)), "levels should be a set or list"
    assert len(levels) > 0, "levels should not be empty"
    
    # Check content format (tuple of ints)
    first_level = list(levels)[0]
    assert isinstance(first_level, tuple), "Level items should be tuples"
    assert len(first_level) == 2, "Level tuples should have 2 elements (world, stage)"
    
    print(f"Verified info format: {info['retro_episode']}")

if __name__ == "__main__":
    test_mario_episode_info_format()
