import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from typing import Optional
from utils import SmallConvNet, SmallDeconvNet, flatten_two_dims, unflatten_first_dim, init_weights_fc, activ
from cnn_policy import CnnPolicy

class FeatureExtractor(nn.Module):
    def __init__(self, policy: CnnPolicy, features_shared_with_policy: bool, feat_dim: Optional[int] = None, layernormalize: Optional[bool] = None, scope: str = 'feature_extractor'):
        super().__init__() # type:ignore
        self.scope = scope
        self.features_shared_with_policy = features_shared_with_policy
        self.feat_dim = feat_dim
        self.layernormalize = layernormalize
        self.policy = policy
        self.hidsize = policy.hidsize
        self.ob_space = policy.ob_space
        self.ac_space = policy.ac_space
        
        # We expect policy to have ob_mean/std as numpy arrays or tensors
        self.register_buffer('ob_mean', torch.as_tensor(policy.ob_mean).float())
        self.register_buffer('ob_std', torch.as_tensor(policy.ob_std).float())

        if features_shared_with_policy:
            self.feature_net = self.policy.feature_net
        else:
            # Detect input channels from ob_space
            # Gym spaces are usually (H, W, C). We need C.
            in_channels = self.ob_space.shape[-1] # type: ignore
            self.feature_net = SmallConvNet(in_channels, feat_dim, activ, None, layernormalize) # type: ignore

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) or (B, C, H, W)
        x_has_timesteps = (x.dim() == 5)
        
        if x_has_timesteps:
            sh = x.shape
            x = flatten_two_dims(x)
        
        mean = self.ob_mean
        assert mean.shape == x.shape[-3:]
        std = self.ob_std
        
        x = (x.float() - mean) / std
        x = self.feature_net(x)
        
        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def get_loss(self, obs: torch.Tensor, last_obs: torch.Tensor, acs: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0, device=obs.device)


class InverseDynamics(FeatureExtractor):
    def __init__(self, policy: CnnPolicy, features_shared_with_policy: bool, feat_dim: int, layernormalize: bool):
        super(InverseDynamics, self).__init__(scope="inverse_dynamics", policy=policy,
                                              features_shared_with_policy=features_shared_with_policy,
                                              feat_dim=feat_dim, layernormalize=layernormalize)
        
        # Inverse Dynamics MLP: (feat + next_feat) -> hid -> ac
        input_dim = self.feat_dim * 2 # Concatenation of phi(s) and phi(s') # type: ignore
        
        self.fc1 = nn.Linear(input_dim, self.policy.hidsize)
        self.fc2 = nn.Linear(self.policy.hidsize, self.ac_space.n) # type: ignore
        
        init_weights_fc(self.fc1)
        init_weights_fc(self.fc2)

    def get_loss(self, obs: torch.Tensor, last_obs: torch.Tensor, acs: torch.Tensor) -> torch.Tensor:
        # obs: (B, T, C, H, W)
        # last_obs: (B, 1, C, H, W)
        # acs: (B, T)
        
        features = self.get_features(obs) # (B, T, feat_dim)
        last_features = self.get_features(last_obs) # (B, 1, feat_dim)
        
        # Construct next_features aligned with features
        # obs covers t=0...T-1. next_obs should cover t=1...T
        # features[:, 1:] is 1...T-1.
        # We append last_features (T).
        
        next_features = torch.cat([features[:, 1:], last_features], dim=1)
        
        # Input to IDM
        x = torch.cat([features, next_features], dim=2) # (B, T, 2*feat_dim)
        x = flatten_two_dims(x)
        
        x = activ(self.fc1(x))
        logits = self.fc2(x)
        
        # Loss
        # acs is (B, T). Flatten to (B*T)
        acs_flat = acs.view(-1)
        return F.cross_entropy(logits, acs_flat.long())


class VAE(FeatureExtractor):
    def __init__(self, policy: CnnPolicy, features_shared_with_policy: bool, feat_dim: int, layernormalize: bool = False, spherical_obs: bool = False):
        assert not layernormalize, "VAE features should already have reasonable size, no need to layer normalize them"
        self.spherical_obs = spherical_obs
        
        # Note: VAE feat_dim is effectively doubled in get_features in original code to split into mean/scale.
        # So we pass feat_dim to super, but inside we might need to adjust logic.
        # Actually, TF code: "feat_dim=2 * self.feat_dim" passed to small_convnet.
        # Here we need to instruct SmallConvNet to output 2*feat_dim.
        
        # We cannot easily change the SmallConvNet output size if shared. 
        # But VAE usually implies specific features. 
        # "features_shared_with_policy" logic in __init__:
        # TF: self.features = split(self.features)[0] if shared.
        
        super(VAE, self).__init__(scope="vae", policy=policy,
                                  features_shared_with_policy=features_shared_with_policy,
                                  feat_dim=feat_dim, layernormalize=False)
        
        # If not shared, we need our own net with double output
        if not features_shared_with_policy:
            in_channels = self.ob_space.shape[-1] # type: ignore
            self.feature_net = SmallConvNet(in_channels, 2 * feat_dim, activ, None, layernormalize=False)
        
        # Decoder
        # Input: feat_dim (sampled z). Output: distribution param (loc, scale)
        # Output channels: 4 if spherical (just mean), else 8 (mean + scale)
        out_ch = 4 if self.spherical_obs else 8 # Assuming 4 stack? 
        # ob_space.shape[-1] is the channel count.
        self.obs_channels = self.ob_space.shape[-1] # type: ignore
        dec_out_ch = self.obs_channels if self.spherical_obs else 2 * self.obs_channels
        
        self.decoder_net = SmallDeconvNet(feat_dim, activ, dec_out_ch, positional_bias=True)
        
        if self.spherical_obs:
            self.scale = nn.Parameter(torch.ones(1)) 

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        # Override to handle the double dimension
        # Normalize
        mean = self.ob_mean
        std = self.ob_std
        assert mean.shape == x.shape[-3:]

        x_has_timesteps = (x.dim() == 5)
        if x_has_timesteps:
            sh = x.shape
            x = flatten_two_dims(x)
        
        x_norm = (x.float() - mean) / std
        features = self.feature_net(x_norm)
        
        if x_has_timesteps:
            features = unflatten_first_dim(features, sh)
        
        return features

    def get_loss(self, obs: torch.Tensor, last_obs: torch.Tensor, acs: Optional[torch.Tensor] = None) -> torch.Tensor:
        # obs: (B, T, C, H, W)
        features = self.get_features(obs) # (B, T, 2*feat_dim)
        
        posterior_mean, posterior_scale = torch.chunk(features, 2, dim=-1)
        posterior_scale = F.softplus(posterior_scale)
        posterior_dist = dist.Normal(posterior_mean, posterior_scale)
        
        # Prior: N(0, 1)
        prior_mean = torch.zeros_like(posterior_mean)
        prior_scale = torch.ones_like(posterior_scale)
        prior_dist = dist.Normal(prior_mean, prior_scale)
        
        posterior_kl = dist.kl_divergence(posterior_dist, prior_dist)
        posterior_kl = torch.sum(posterior_kl, dim=-1) # Sum over feature dim
        
        posterior_sample = posterior_dist.rsample()
        
        # Decoder
        # Flatten sample for decoder
        sh = posterior_sample.shape
        z = flatten_two_dims(posterior_sample)
        
        recon_out = self.decoder_net(z)
        
        if self.spherical_obs:
            recon_mean = recon_out
            scale_val = torch.max(self.scale, torch.tensor(-4.0, device=self.scale.device))
            scale_val = F.softplus(scale_val)
            recon_scale = scale_val * torch.ones_like(recon_mean)
        else:
            recon_mean, recon_scale = torch.chunk(recon_out, 2, dim=1) # dim 1 is channel in NCHW
            recon_scale = F.softplus(recon_scale)
            
        recon_dist = dist.Normal(recon_mean, recon_scale)
        
        # Prepare targets
        # Flatten obs: (B*T, C, H, W)
        target = flatten_two_dims(obs)
        target_norm = self.add_noise_and_normalize(target)
        
        # Likelihood
        log_prob = recon_dist.log_prob(target_norm)
        # Sum over spatial dims and channels: (B*T, C, H, W) -> (B*T)
        log_prob = torch.sum(log_prob, dim=[1, 2, 3])
        
        # Reshape to (B, T)
        reconstruction_likelihood = unflatten_first_dim(log_prob, sh)
        
        likelihood_lower_bound = reconstruction_likelihood - posterior_kl
        return -likelihood_lower_bound

    def add_noise_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, C, H, W) or (B, T, C, H, W)
        # Add uniform noise [0, 1]
        noise = torch.rand_like(x.float())
        x = x.float() + noise
        
        mean = self.ob_mean
        std = self.ob_std
             
        return (x - mean) / std


class JustPixels(FeatureExtractor):
    def __init__(self, policy: CnnPolicy, features_shared_with_policy: bool, feat_dim: Optional[int] = None, layernormalize: Optional[bool] = None, scope: str = 'just_pixels'):
        assert not layernormalize
        assert not features_shared_with_policy
        super(JustPixels, self).__init__(scope=scope, policy=policy,
                                         features_shared_with_policy=False,
                                         feat_dim=None, layernormalize=None)
        # No feature net needed

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        # Just normalize
        # x: (B, T, C, H, W) or (B, C, H, W)
        mean = self.ob_mean
        std = self.ob_std
        
        return (x.float() - mean) / std

    def get_loss(self, obs: torch.Tensor, last_obs: torch.Tensor, acs: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0, device=obs.device)