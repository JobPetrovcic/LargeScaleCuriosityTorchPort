import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union, Any, Dict, Callable

from auxiliary_tasks import JustPixels, FeatureExtractor
from utils import SmallConvNet, UNet, flatten_two_dims, unflatten_first_dim, activ, init_weights_fc, init_weights_conv

class DynamicsDenseBlock(nn.Module):
    """
    Helper block for the dense dynamics model.
    Corresponds to: tf.layers.dense(add_ac(x), ...)
    """
    def __init__(self, input_dim: int, ac_dim: int, out_dim: int, activation: Optional[Callable] = None):
        super().__init__() # type:ignore
        self.fc = nn.Linear(input_dim + ac_dim, out_dim)
        self.activation = activation
        init_weights_fc(self.fc)

    def forward(self, x: torch.Tensor, ac_one_hot: torch.Tensor) -> torch.Tensor:
        # x: (B, D), ac: (B, A)
        inputs = torch.cat([x, ac_one_hot], dim=1)
        out = self.fc(inputs)
        if self.activation:
            out = self.activation(out)
        return out

class DynamicsResidualBlock(nn.Module):
    """
    Corresponds to the residual loop in Dynamics.get_loss:
    res = dense(add_ac(x), hid, leaky)
    res = dense(add_ac(res), hid, None)
    return x + res
    """
    def __init__(self, hid_size: int, ac_dim: int):
        super().__init__() # type:ignore
        self.dense1 = DynamicsDenseBlock(hid_size, ac_dim, hid_size, activation=activ)
        self.dense2 = DynamicsDenseBlock(hid_size, ac_dim, hid_size, activation=None)

    def forward(self, x: torch.Tensor, ac_one_hot: torch.Tensor) -> torch.Tensor:
        res = self.dense1(x, ac_one_hot)
        res = self.dense2(res, ac_one_hot)
        return x + res

class Dynamics(nn.Module):
    def __init__(self, auxiliary_task: FeatureExtractor, predict_from_pixels: bool, feat_dim: int, scope: str = 'dynamics'):
        super().__init__() # type:ignore
        self.scope = scope
        self.auxiliary_task = auxiliary_task
        self.hidsize = self.auxiliary_task.hidsize
        self.feat_dim = feat_dim
        self.predict_from_pixels = predict_from_pixels
        
        # We need the action space size for one-hot encoding
        self.ac_space_n = self.auxiliary_task.ac_space.n # type: ignore
        
        # If predict_from_pixels, we have our own feature extractor.
        # Otherwise we use the aux task's features.
        if self.predict_from_pixels:
            in_channels = self.auxiliary_task.ob_space.shape[-1] # type: ignore
            self.feature_net = SmallConvNet(in_channels, feat_dim, activ, activ, layernormalize=False) # type: ignore
        else:
            self.feature_net = None

        # Build the Dynamics Network (MLP)
        # Structure from TF:
        # x = dense(add_ac(features), hid, leaky)
        # 4x residual(x)
        # x = dense(add_ac(x), out_dim, None)
        
        # Input dim is feat_dim (if from pixels or aux) 
        # But wait, aux task features might be different size?
        # auxiliary_tasks.py: "FeatureExtractor ... feat_dim=512".
        # VAE uses 2*feat_dim internally but exposes split features? 
        # VAE init: "self.features = tf.split(self.features, 2, -1)[0]"
        # So effective input dim is feat_dim.
        
        input_dim = self.feat_dim
        
        self.first_dense = DynamicsDenseBlock(input_dim, self.ac_space_n, self.hidsize, activation=activ)
        self.res_blocks = nn.ModuleList([DynamicsResidualBlock(self.hidsize, self.ac_space_n) for _ in range(4)])
        self.last_dense = DynamicsDenseBlock(self.hidsize, self.ac_space_n, input_dim, activation=None)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        # Only used if predict_from_pixels is True
        # x: (B, T, C, H, W) or (B, C, H, W)
        x_has_timesteps = (x.dim() == 5)
        if x_has_timesteps:
            sh = x.shape
            x = flatten_two_dims(x)
        
        # Normalize using aux task stats
        mean = self.auxiliary_task.ob_mean
        std = self.auxiliary_task.ob_std
        
        x = (x.float() - mean) / std
        x = self.feature_net(x) # type: ignore
        
        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def forward(self, features: torch.Tensor, ac_one_hot: torch.Tensor) -> torch.Tensor:
        # features: (B, D)
        # ac_one_hot: (B, A)
        x = self.first_dense(features, ac_one_hot)
        for block in self.res_blocks:
            x = block(x, ac_one_hot)
        x = self.last_dense(x, ac_one_hot)
        return x

    def get_loss(self, obs: torch.Tensor, last_obs: torch.Tensor, acs: torch.Tensor) -> torch.Tensor:
        # obs: (B, T, C, H, W)
        # acs: (B, T)
        
        if self.predict_from_pixels:
            features = self.get_features(obs)
        else:
            # Detach features from auxiliary task! 
            # "self.features = tf.stop_gradient(self.auxiliary_task.features)"
            features = self.auxiliary_task.get_features(obs).detach()
            
            if hasattr(self.auxiliary_task, 'spherical_obs'): # It is VAE
                features = torch.chunk(features, 2, dim=-1)[0]

        if self.predict_from_pixels:
            pass
        
        target_features_all = self.auxiliary_task.get_features(obs)
        target_last_features = self.auxiliary_task.get_features(last_obs)
        
        # Handle VAE split for targets too
        if hasattr(self.auxiliary_task, 'spherical_obs'):
            target_features_all = torch.chunk(target_features_all, 2, dim=-1)[0]
            target_last_features = torch.chunk(target_last_features, 2, dim=-1)[0]
            
        target_features = torch.cat([target_features_all[:, 1:], target_last_features], dim=1)
        target_features = target_features.detach()

        sh = features.shape
        x = flatten_two_dims(features) # (B*T, D)
        
        # Actions
        acs_flat = acs.view(-1).long()
        ac_one_hot = F.one_hot(acs_flat, self.ac_space_n).float()
        
        # Forward
        predicted_next_features = self.forward(x, ac_one_hot)
         
        target_flat = flatten_two_dims(target_features)
        mse = torch.mean((predicted_next_features - target_flat) ** 2, dim=-1)
        
        return unflatten_first_dim(mse, sh)

    def calculate_loss(self, ob: np.ndarray[Any, Any], last_ob: np.ndarray[Any, Any], acs: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        # Used for intrinsic reward calculation.
        # ob: numpy array (N, T, H, W, C) -> Need to transpose to NCHW
        # acs: numpy array
        
        # TF logic: n_chunks=8. 
        n_chunks = 8
        n = ob.shape[0]
        chunk_size = n // n_chunks
        assert n % n_chunks == 0
        
        # Convert inputs to torch
        # Note: input ob is (N, T, H, W, C).
        # We need to permute to (N, T, C, H, W).
        
        def to_torch(arr: np.ndarray[Any, Any]) -> torch.Tensor:
            t = torch.tensor(arr).to(self.auxiliary_task.ob_mean.device)
            if t.dim() == 5: # (N, T, H, W, C)
                t = t.permute(0, 1, 4, 2, 3)
            elif t.dim() == 4: # (N, H, W, C) - last_ob
                t = t.permute(0, 3, 1, 2)
            assert t.shape[-1] == t.shape[-2]
            return t.float()

        loss_list = []
        
        # No grad for inference
        with torch.no_grad():
            for i in range(n_chunks):
                sli = slice(i * chunk_size, (i + 1) * chunk_size)
                
                ob_chunk = to_torch(ob[sli])
                last_ob_chunk = to_torch(last_ob[sli])
                acs_chunk = torch.tensor(acs[sli]).to(self.auxiliary_task.ob_mean.device)
                
                # get_loss returns (Batch, Time)
                l = self.get_loss(ob_chunk, last_ob_chunk, acs_chunk)
                loss_list.append(l.cpu().numpy())
                
        return np.concatenate(loss_list, axis=0)


class UNet(Dynamics):
    def __init__(self, auxiliary_task: FeatureExtractor, predict_from_pixels: bool, feat_dim: int, scope: str = 'pixel_dynamics'):
        assert isinstance(auxiliary_task, JustPixels)
        assert not predict_from_pixels
        super(UNet, self).__init__(auxiliary_task=auxiliary_task,
                                   predict_from_pixels=predict_from_pixels,
                                   feat_dim=feat_dim,
                                   scope=scope)
        
        # Override the network with UNet
        in_channels = self.auxiliary_task.ob_space.shape[-1] # type: ignore
        self.unet = UNetNetwork(in_channels, feat_dim, activ) # defined in utils as UNet
        
        # TF logic stores a prediction_pixels variable. 
        # We'll need to replicate that if it's used elsewhere, but likely only for viz.

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, features: torch.Tensor, ac_one_hot: torch.Tensor) -> torch.Tensor:
        # UNet forward. 
        # Features here are the pixels (normalized).
        return self.unet(features, ac_one_hot)

    def get_loss(self, obs: torch.Tensor, last_obs: torch.Tensor, acs: torch.Tensor) -> torch.Tensor:
        # obs: (B, T, C, H, W)
        
        # Features = normalized pixels
        features = self.auxiliary_task.get_features(obs).detach()
        
        # Target = next normalized pixels
        target_features_all = self.auxiliary_task.get_features(obs)
        target_last_features = self.auxiliary_task.get_features(last_obs)
        target_features = torch.cat([target_features_all[:, 1:], target_last_features], dim=1)
        target_features = target_features.detach()
        
        sh = features.shape
        
        # Flatten
        x = flatten_two_dims(features)
        
        # Actions
        acs_flat = acs.view(-1).long()
        ac_one_hot = F.one_hot(acs_flat, self.ac_space_n).float()
        
        predicted_pixels = self.forward(x, ac_one_hot)
        loss = torch.mean((predicted_pixels - flatten_two_dims(target_features)) ** 2, dim=[1, 2, 3])
        
        # Unflatten to (B, T)
        return unflatten_first_dim(loss, sh)

# Alias for utils.UNet to avoid name collision with class UNet
from utils import UNet as UNetNetwork