
import pytest
import numpy as np
import torch
import gym
from utils import random_agent_ob_mean_std, SmallConvNet, UNet, activ, Conv2dSame
from typing import Any

class DummyEnv(gym.Env[Any, Any]):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(4)
    
    def reset(self):
        return np.zeros((84, 84, 1), dtype=np.uint8)
    
    def step(self, action: Any):
        obs = np.random.randint(0, 255, (84, 84, 1), dtype=np.uint8)
        return obs, 0.0, False, {}

def test_random_agent_ob_mean_std():
    env = DummyEnv()
    nsteps = 100
    mean, std = random_agent_ob_mean_std(env, nsteps=nsteps)
    
    # Check shapes
    # mean should be (H, W, C) based on np.mean(obs_arr, axis=0)
    assert isinstance(mean, np.ndarray)
    assert mean.shape == (1, 84, 84)
    
    # std should be a scalar float based on np.std(obs_arr, axis=0).mean()
    assert isinstance(std, (float, np.float32, np.float64))

def test_small_conv_net_input_shape():
    in_channels = 1
    feat_dim = 256
    net = SmallConvNet(in_channels, feat_dim, activ, None, False)
    
    # Correct shape: (Batch, Channels, Height, Width)
    # SmallConvNet expects 84x84 input due to hardcoded linear layer size 3136 (7*7*64)
    # 84 -> 21 -> 10 -> 7 (strides 4, 2, 1 with kernel sizes 8, 4, 3? Let's check logic)
    # Conv1: 84 -> (84-8)/4 + 1 = 20? No, pytorch default padding is 0.
    # Let's check SmallConvNet definition in utils.py
    # conv1: 8x8, stride 4. (84-8)/4 + 1 = 19 + 1 = 20.
    # conv2: 4x4, stride 2. (20-4)/2 + 1 = 8 + 1 = 9.
    # conv3: 3x3, stride 1. (9-3)/1 + 1 = 7.
    # 7*7*64 = 3136. Correct.
    
    x_correct = torch.randn(2, 1, 84, 84)
    out = net(x_correct)
    assert out.shape == (2, feat_dim)
    
    # Incorrect channel size
    x_wrong_channel = torch.randn(2, 3, 84, 84)
    with pytest.raises(AssertionError, match="Input channel mismatch"):
        net(x_wrong_channel)

def test_unet_input_shape():
    in_channels = 1
    feat_dim = 256
    unet = UNet(in_channels, feat_dim, activ)
    
    # UNet uses Conv2dSame, so spatial dims are preserved/strided predictably.
    # It expects (B, C, H, W)
    
    x_correct = torch.randn(2, 1, 84, 84)
    ac_one_hot = torch.randn(2, 4) # Dummy action
    
    # Initialize layers first call
    out = unet(x_correct, ac_one_hot)
    # Output shape depends on UNet architecture, usually same spatial as input or similar
    # The code says: out = out[:, :, 6:-6, 6:-6] at the end.
    # And it pads input by (6,6,6,6).
    # So output should be roughly input size.
    assert out.shape[2] == 84
    assert out.shape[3] == 84
    
    # Incorrect channel size
    x_wrong_channel = torch.randn(2, 3, 84, 84)
    with pytest.raises(AssertionError, match="Input channel mismatch"):
        unet(x_wrong_channel, ac_one_hot)

def test_conv2d_same():
    # Test stride 1 (should preserve size)
    conv = Conv2dSame(in_channels=1, out_channels=1, kernel_size=3, stride=1)
    x = torch.randn(1, 1, 10, 10)
    out = conv(x)
    assert out.shape == (1, 1, 10, 10)

    # Test stride 2 (should halve size)
    conv = Conv2dSame(in_channels=1, out_channels=1, kernel_size=3, stride=2)
    x = torch.randn(1, 1, 10, 10)
    out = conv(x)
    assert out.shape == (1, 1, 5, 5)
    
    # Test with odd input size and stride 2
    x = torch.randn(1, 1, 11, 11)
    out = conv(x)
    # ceil(11/2) = 6
    assert out.shape == (1, 1, 6, 6)
