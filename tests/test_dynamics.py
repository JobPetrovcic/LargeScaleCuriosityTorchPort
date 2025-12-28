
from typing import Tuple
import torch
import torch.nn as nn
import numpy as np
import gym
from dynamics import DynamicsDenseBlock, DynamicsResidualBlock, Dynamics
from auxiliary_tasks import InverseDynamics
from utils import activ

class MockAuxiliaryTask(nn.Module):
    def __init__(self, hidsize: int, ac_space_n: int, ob_shape: Tuple[int, ...], feat_dim: int = 128):
        super().__init__() # type:ignore
        self.hidsize = hidsize
        self.ac_space = type('obj', (object,), {'n': ac_space_n})
        # ob_shape is (C, H, W), but code expects (H, W, C) in ob_space
        self.ob_space = type('obj', (object,), {'shape': (ob_shape[1], ob_shape[2], ob_shape[0])})
        self.ob_mean = torch.zeros(ob_shape)
        self.ob_std = torch.ones(ob_shape)
        self.feat_dim = feat_dim
        
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        # Return dummy features (B, T, D) or (B, D)
        # x is (B, T, C, H, W) or (B, C, H, W)
        if x.dim() == 5:
            return torch.randn(x.shape[0], x.shape[1], self.feat_dim)
        else:
            return torch.randn(x.shape[0], self.feat_dim)

class MockPolicy:
    def __init__(self):
        self.hidsize = 64
        self.ob_space = gym.spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8)
        self.ac_space = gym.spaces.Discrete(4)
        self.ob_mean = np.zeros((3, 84, 84), dtype=np.float32)
        self.ob_std = np.ones((3, 84, 84), dtype=np.float32)
        self.feature_net = None 

def test_reshaping():
    x = torch.randn(5, 3, 84, 84)
    mean = torch.zeros(3, 84, 84)

    assert torch.allclose(x - mean, x - mean.unsqueeze(0))

def test_dynamics_dense_block():
    input_dim = 32
    ac_dim = 4
    out_dim = 64
    batch_size = 10
    
    block = DynamicsDenseBlock(input_dim, ac_dim, out_dim, activation=activ)
    
    x = torch.randn(batch_size, input_dim)
    ac_one_hot = torch.randn(batch_size, ac_dim)
    
    out = block(x, ac_one_hot)
    assert out.shape == (batch_size, out_dim)

def test_dynamics_residual_block():
    hid_size = 32
    ac_dim = 4
    batch_size = 10
    
    block = DynamicsResidualBlock(hid_size, ac_dim)
    
    x = torch.randn(batch_size, hid_size)
    ac_one_hot = torch.randn(batch_size, ac_dim)
    
    out = block(x, ac_one_hot)
    assert out.shape == (batch_size, hid_size)

def test_dynamics_init_and_shapes():
    feat_dim = 128
    hidsize = 64
    ac_space_n = 4
    ob_shape = (3, 84, 84) # C, H, W (Gym standard)
    
    aux_task = MockAuxiliaryTask(hidsize, ac_space_n, ob_shape, feat_dim=feat_dim)
    
    # Test with predict_from_pixels=True
    dynamics = Dynamics(aux_task, predict_from_pixels=True, feat_dim=feat_dim)
    
    assert dynamics.feature_net is not None
    
    # Test get_features with pixels
    batch_size = 5
    # Input to get_features is usually (B, C, H, W) or (B, T, C, H, W)
    # So we need to permute ob_shape for input x
    x = torch.randn(batch_size, 3, 84, 84) # (B, C, H, W)
    
    # Mock aux task mean/std to be compatible
    aux_task.ob_mean = torch.zeros(ob_shape) # (C, H, W)
    aux_task.ob_std = torch.ones(ob_shape)
    
    features = dynamics.get_features(x)
    assert features.shape == (batch_size, feat_dim)
    
    # Test with predict_from_pixels=False
    dynamics_no_pix = Dynamics(aux_task, predict_from_pixels=False, feat_dim=feat_dim)
    assert dynamics_no_pix.feature_net is None

def test_inverse_dynamics_loss():
    policy = MockPolicy()
    feat_dim = 32
    
    inv_dyn = InverseDynamics(policy, features_shared_with_policy=False, feat_dim=feat_dim, layernormalize=False)
    
    B, T = 2, 5
    obs = torch.randn(B, T, 3, 84, 84)
    last_obs = torch.randn(B, 1, 3, 84, 84)
    acs = torch.randint(0, 4, (B, T))
    
    loss = inv_dyn.get_loss(obs, last_obs, acs)
    
    # InverseDynamics.get_loss returns cross_entropy loss (scalar)
    assert loss.dim() == 0
    assert loss.item() >= 0

def test_dynamics_forward_full():
    feat_dim = 32
    hidsize = 64
    ac_space_n = 4
    ob_shape = (5, 84, 84)
    
    aux_task = MockAuxiliaryTask(hidsize, ac_space_n, ob_shape, feat_dim=feat_dim)
    
    dyn = Dynamics(aux_task, predict_from_pixels=False, feat_dim=feat_dim)
    
    B = 2
    features = torch.randn(B, feat_dim)
    ac_one_hot = torch.randn(B, ac_space_n)
    
    out = dyn.forward(features, ac_one_hot)
    assert out.shape == (B, feat_dim)

def test_dynamics_loss():
    feat_dim = 32
    hidsize = 64
    ac_space_n = 4
    ob_shape = (3, 84, 84)
    
    aux_task = MockAuxiliaryTask(hidsize, ac_space_n, ob_shape, feat_dim=feat_dim)
    
    dyn = Dynamics(aux_task, predict_from_pixels=False, feat_dim=feat_dim)
    
    B, T = 2, 5
    obs = torch.randn(B, T, 3, 84, 84)
    last_obs = torch.randn(B, 1, 3, 84, 84)
    acs = torch.randint(0, 4, (B, T))
    
    loss = dyn.get_loss(obs, last_obs, acs)
    
    # Dynamics.get_loss returns (B, T) tensor of MSE errors
    assert loss.shape == (B, T)
    assert (loss >= 0).all()

def test_dynamics_predict_from_pixels_loss():
    feat_dim = 32
    hidsize = 64
    ac_space_n = 4
    ob_shape = (3, 84, 84)
    
    aux_task = MockAuxiliaryTask(hidsize, ac_space_n, ob_shape, feat_dim=feat_dim)
    
    # When predict_from_pixels=True, Dynamics creates its own feature_net
    dyn = Dynamics(aux_task, predict_from_pixels=True, feat_dim=feat_dim)
    
    B, T = 2, 5
    obs = torch.randn(B, T, 3, 84, 84)
    last_obs = torch.randn(B, 1, 3, 84, 84)
    acs = torch.randint(0, 4, (B, T))
    
    loss = dyn.get_loss(obs, last_obs, acs)
    
    assert loss.shape == (B, T)