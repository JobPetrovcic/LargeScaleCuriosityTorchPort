
import gym
import numpy as np
import pytest
from wrappers import MaxAndSkipEnv, ProcessFrame84, ExtraTimeLimit, AddRandomStateToInfo, FrameSkip, OneChannel
from monitor import Monitor
from typing import Any

class DummyEnv(gym.Env[Any, Any]):
    def __init__(self, return_info_on_reset=True):
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(2)
        self.steps = 0
        self.return_info_on_reset = return_info_on_reset

    def reset(self, **kwargs):
        self.steps = 0
        obs = np.zeros((210, 160, 3), dtype=np.uint8)
        if self.return_info_on_reset:
            return obs, {}
        return obs

    def step(self, action):
        self.steps += 1
        done = self.steps >= 10
        # Return 5-tuple (obs, reward, terminated, truncated, info) to match modern gym expectations in wrappers
        # But wait, the wrappers handle both. Let's return 5-tuple as it's more robust for modern gym checks.
        return np.zeros((210, 160, 3), dtype=np.uint8), 1.0, done, False, {}

def test_monitor_wrapper_adds_episode_info():
    env = DummyEnv()
    env = Monitor(env)
    env.reset()
    for _ in range(10):
        # Monitor wrapper handles 5-tuple or 4-tuple from inner env and returns 5-tuple or 4-tuple
        # Our DummyEnv returns 5-tuple.
        step_result = env.step(0)
        if len(step_result) == 5:
            _, _, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            _, _, done, info = step_result
            
        if done:
            assert "episode" in info, "Monitor wrapper must add 'episode' to info on done"
            assert "r" in info["episode"]
            assert "l" in info["episode"]
            assert info["episode"]["r"] == 10.0
            assert info["episode"]["l"] == 10

def test_max_and_skip_env():
    env = DummyEnv()
    skip = 4
    env = MaxAndSkipEnv(env, skip=skip)
    env.reset()
    
    # Step once, should advance underlying env by 'skip' steps
    step_result = env.step(0)
    if len(step_result) == 5:
        _, reward, _, _, _ = step_result
    else:
        _, reward, _, _ = step_result
        
    assert env.env.steps == skip
    assert reward == skip * 1.0 # 1.0 reward per step

def test_process_frame84():
    env = DummyEnv()
    env = ProcessFrame84(env, crop=True)
    obs, _ = env.reset()
    assert obs.shape == (84, 84, 1)
    
    step_result = env.step(0)
    obs = step_result[0]
    assert obs.shape == (84, 84, 1)

def test_frame_skip():
    env = DummyEnv()
    skip = 3
    env = FrameSkip(env, n=skip)
    env.reset()
    
    step_result = env.step(0)
    if len(step_result) == 5:
        _, reward, _, _, _ = step_result
    else:
        _, reward, _, _ = step_result
        
    assert env.env.steps == skip
    assert reward == skip * 1.0

def test_extra_time_limit():
    env = DummyEnv()
    max_steps = 5
    env = ExtraTimeLimit(env, max_episode_steps=max_steps)
    env.reset()
    
    for i in range(max_steps):
        step_result = env.step(0)
        if len(step_result) == 5:
            _, _, term, trunc, _ = step_result
            done = term or trunc
        else:
            _, _, done, _ = step_result
            
        if i < max_steps - 1:
            assert not done
        else:
            assert done
            # Check if it was truncated (if 5-tuple)
            if len(step_result) == 5:
                assert step_result[3] == True # truncated

