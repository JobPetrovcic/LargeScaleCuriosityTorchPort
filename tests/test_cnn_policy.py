
import pytest
import numpy as np
import torch
import gym
from cnn_policy import CnnPolicy
from utils import activ

def test_cnn_policy_init():
    ob_space = gym.spaces.Box(0, 255, (84, 84, 5), dtype=np.uint8)
    ac_space = gym.spaces.Discrete(4)
    hidsize = 64
    feat_dim = 32
    ob_mean = np.zeros((5, 84, 84), dtype=np.float32)
    ob_std = np.ones((5, 84, 84), dtype=np.float32)
    
    policy = CnnPolicy(ob_space, ac_space, hidsize, ob_mean, ob_std, feat_dim, 
                       layernormalize=False, nl=activ)
    
    assert policy.feature_net is not None
    assert policy.fc1 is not None
    assert policy.fc2 is not None
    assert policy.pd_head is not None
    assert policy.v_head is not None

def test_cnn_policy_forward():
    ob_space = gym.spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8)
    ac_space = gym.spaces.Discrete(4)
    hidsize = 64
    feat_dim = 32
    ob_mean = np.zeros((3, 84, 84), dtype=np.float32)
    ob_std = np.ones((3, 84, 84), dtype=np.float32)
    
    policy = CnnPolicy(ob_space, ac_space, hidsize, ob_mean, ob_std, feat_dim, 
                       layernormalize=False, nl=activ)
    
    B, T = 2, 5
    # Input shape: (B, T, C, H, W)
    # ob_space is (H, W, C). CnnPolicy expects (C, H, W) in tensor but handles permute in get_features?
    # No, get_features expects (B, T, C, H, W) or (B, C, H, W).
    # And it permutes ob_mean/std if needed.
    # So we should pass (B, T, C, H, W).
    
    obs = torch.randn(B, T, 3, 84, 84)
    
    a_samp, vpred, nlp, entropy, pdparam = policy.forward(obs)
    
    assert a_samp.shape == (B, T)
    assert vpred.shape == (B, T)
    assert nlp.shape == (B, T)
    assert entropy.shape == (B, T)
    assert pdparam.shape == (B, T, ac_space.n)

def test_cnn_policy_get_features():
    ob_space = gym.spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8)
    ac_space = gym.spaces.Discrete(4)
    hidsize = 64
    feat_dim = 32
    ob_mean = np.zeros((3, 84, 84), dtype=np.float32)
    ob_std = np.ones((3, 84, 84), dtype=np.float32)
    
    policy = CnnPolicy(ob_space, ac_space, hidsize, ob_mean, ob_std, feat_dim, 
                       layernormalize=False, nl=activ)
    
    B, T = 2, 5
    obs = torch.randn(B, T, 3, 84, 84)
    
    features = policy.get_features(obs)
    assert features.shape == (B, T, feat_dim)
    
    # Test without time dim
    obs_no_time = torch.randn(B, 3, 84, 84)
    features_no_time = policy.get_features(obs_no_time)
    assert features_no_time.shape == (B, feat_dim)
    
def test_cnn_policy_get_ac_value_nlp():
    ob_space = gym.spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8)
    ac_space = gym.spaces.Discrete(4)
    hidsize = 64
    feat_dim = 32
    ob_mean = np.zeros((3, 84, 84), dtype=np.float32)
    ob_std = np.ones((3, 84, 84), dtype=np.float32)
    
    policy = CnnPolicy(ob_space, ac_space, hidsize, ob_mean, ob_std, feat_dim, 
                       layernormalize=False, nl=activ)
    
    # Input is numpy (B, H, W, C)
    B = 2
    obs = np.random.randint(0, 255, (B, 3, 84, 84), dtype=np.uint8)
    
    a_samp, vpred, nlp = policy.get_ac_value_nlp(obs)
    
    assert a_samp.shape == (B,)
    assert vpred.shape == (B,)
    assert nlp.shape == (B,)
