import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
import gym
from typing import Tuple, Optional, Any, Dict, Callable
from utils import SmallConvNet, flatten_two_dims, unflatten_first_dim, init_weights_fc, activ

class CnnPolicy(nn.Module):
    def __init__(self, ob_space: gym.Space[Any], ac_space: gym.Space[Any], hidsize: int,
                 ob_mean: np.ndarray[Any, Any], ob_std: np.ndarray[Any, Any], feat_dim: int, layernormalize: bool, nl: Callable, scope: str = "policy"):
        super().__init__() # type:ignore
        self.layernormalize = layernormalize
        self.nl = nl
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.hidsize = hidsize
        self.feat_dim = feat_dim
        self.scope = scope
        
        # Register normalization constants
        # ob_mean/std are numpy arrays (H, W, C) from utils.random_agent_ob_mean_std
        self.register_buffer('ob_mean', torch.as_tensor(ob_mean).float())
        self.register_buffer('ob_std', torch.as_tensor(ob_std).float())

        # Feature Extractor
        in_channels = ob_space.shape[-1] # type: ignore
        self.feature_net = SmallConvNet(in_channels, feat_dim, nl, None, layernormalize)
        
        # Policy & Value Heads
        # TF: fc(units=hidsize) -> fc(units=hidsize) -> heads
        self.fc1 = nn.Linear(feat_dim, hidsize)
        self.fc2 = nn.Linear(hidsize, hidsize)
        
        # Action distribution head
        if hasattr(ac_space, 'n'):
            self.pdparamsize = ac_space.n # type: ignore
        else:
            raise NotImplementedError("Only Discrete spaces supported based on provided code context.")
            
        self.pd_head = nn.Linear(hidsize, self.pdparamsize)
        self.v_head = nn.Linear(hidsize, 1)
        
        # Init weights
        # utils.fc uses normc_initializer(1.)
        init_weights_fc(self.fc1)
        init_weights_fc(self.fc2)
        init_weights_fc(self.pd_head)
        init_weights_fc(self.v_head)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) or (B, C, H, W)
        x_has_timesteps = (x.dim() == 5)
        if x_has_timesteps:
            sh = x.shape
            x = flatten_two_dims(x)
        
        mean = self.ob_mean
        std = self.ob_std

        
        x = (x.float() - mean) / std
        x = self.feature_net(x)
        
        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def forward(self, ob: torch.Tensor, ac: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # ob: (B, T, C, H, W)
        # ac: (B, T) optional, for calculating neglogp of specific actions
        
        features = self.get_features(ob)
        
        # Flatten for MLP
        sh = features.shape # (B, T, D)
        x = flatten_two_dims(features)
        
        x = activ(self.fc1(x))
        x = activ(self.fc2(x))
        
        pdparam = self.pd_head(x) # Logits
        vpred = self.v_head(x)
        
        # Unflatten
        pdparam = unflatten_first_dim(pdparam, sh) # (B, T, n_actions)
        vpred = unflatten_first_dim(vpred, sh).squeeze(-1) # (B, T)
        
        # Distribution
        pd = dist.Categorical(logits=pdparam)
        
        # Sample
        a_samp = pd.sample()
        
        # If external actions provided, calc their nlp, else use sampled
        if ac is None:
            ac_ref = a_samp
        else:
            ac_ref = ac
            
        nlp = -pd.log_prob(ac_ref)
        entropy = pd.entropy()
        
        return a_samp, vpred, nlp, entropy, pdparam

    def get_ac_value_nlp(self, ob: np.ndarray[Any, Any]) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        # Interface for rollouts.py
        # ob is numpy array: (Batch, H, W, C) -- Note: Rollouts pass (Batch, H, W, C)
        # But wait, rollouts.py: "self.ph_ob: ob[:, None]" -> Adds time dimension?
        # cnn_policy.py TF: "feed_dict={self.ph_ob: ob[:, None]}"
        # Yes, input is expanded to (Batch, 1, H, W, C).
        
        # Convert to Tensor and Permute
        # Numpy (B, H, W, C) -> Torch (B, C, H, W)
        device = self.ob_mean.device
        ob_tensor = torch.from_numpy(ob).float().to(device)
        
        # Handle Channel First/Last
        # Env gives NHWC. We want NCHW.
        ob_tensor = ob_tensor.unsqueeze(1)

        assert ob_tensor.dim() == 5
        
        # Add Time Dim: (B, 1, C, H, W)
        
        with torch.no_grad():
            a_samp, vpred, nlp, _, _ = self.forward(ob_tensor)
            
        # Return: (Batch, ), (Batch, ), (Batch, )
        # forward returns (Batch, 1).
        return a_samp[:, 0].cpu().numpy(), vpred[:, 0].cpu().numpy(), nlp[:, 0].cpu().numpy()

    def get_var_values(self) -> Dict[str, Any]:
        # For saving/loading state
        return self.state_dict()

    def set_var_values(self, vv: Dict[str, Any]) -> None:
        self.load_state_dict(vv)