import multiprocessing
import os
from typing import List, Tuple, Optional, Union, Any, Callable

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def getdevice() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_mpi_gpus() -> None:
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def guess_available_cpus() -> int:
    return multiprocessing.cpu_count()

def get_mean_and_std(array: np.ndarray[Any, Any]) -> Tuple[float, float]:
    mean = np.mean(array)
    std = np.std(array)
    return mean, std

def random_agent_ob_mean_std(env: gym.Env[Any, Any], nsteps: int = 10000) -> Tuple[float, float]:
    ob = env.reset()
    if isinstance(ob, tuple):
        ob = ob[0]
    obs = [np.asarray(ob)]
    for _ in range(nsteps):
        ac = env.action_space.sample()
        step_result = env.step(ac)
        if len(step_result) == 5:
            ob, _, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            ob, _, done, _ = step_result
            
        if done:
            ob = env.reset()
            if isinstance(ob, tuple):
                ob = ob[0]
        obs.append(np.asarray(ob))
    obs_arr = np.array(obs)
    mean = np.mean(obs_arr, axis=0).astype(np.float32)
    std = np.std(obs_arr, axis=0).mean().astype(np.float32)
    # permute to C,H,W
    mean = np.transpose(mean, (2, 0, 1))    

    return mean, std

def normc_initializer(std: float = 1.0) -> Callable[[torch.Tensor], None]:
    def init(tensor: torch.Tensor) -> None:
        with torch.no_grad():
            tensor.normal_(0, 1)
            tensor.div_(tensor.norm(dim=1, keepdim=True))
            tensor.mul_(std)
    return init

def init_weights_fc(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        normc_initializer(1.0)(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def init_weights_conv(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def activ(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)

def layernorm(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / (torch.sqrt(var) + epsilon)

class LayerNorm(nn.Module):
    def __init__(self, epsilon: float = 1e-8):
        super().__init__() # type:ignore
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layernorm(x, self.epsilon)

class RunningMeanStd(nn.Module):
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        super().__init__() # type:ignore
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float64))
        self.epsilon = epsilon

    def update(self, x: Union[np.ndarray[Any, Any], torch.Tensor]) -> None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.mean.device).double()
        else:
            x = x.double()
        
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        if self.mean.dim() == 0:
            self.mean.data = new_mean
            self.var.data = new_var
            self.count.data = tot_count
        else:
            self.mean[:] = new_mean
            self.var[:] = new_var
            self.count[:] = tot_count

    def forward(self, x: Union[np.ndarray[Any, Any], torch.Tensor]) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.mean.device)
        return (x - self.mean.float()) / (torch.sqrt(self.var.float()) + 1e-8)

def flatten_two_dims(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, *x.shape[2:])

def unflatten_first_dim(x: torch.Tensor, shape_tensor: torch.Tensor) -> torch.Tensor:
    B, T = shape_tensor[0], shape_tensor[1]
    return x.view(B, T, *x.shape[1:])

class Conv2dSame(nn.Conv2d):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return int(max((np.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x=input
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return super().forward(x)

class ConvTranspose2dSame(nn.ConvTranspose2d):
    def forward(self, input: torch.Tensor, output_size: Optional[List[int]] = None) -> torch.Tensor:
        x=input
        out = super().forward(x, output_size)
        _, _, h, w = x.shape
        target_h = h * self.stride[0]
        target_w = w * self.stride[1]
        oh, ow = out.shape[-2:]
        if oh > target_h or ow > target_w:
            crop_h = (oh - target_h) // 2
            crop_w = (ow - target_w) // 2
            out = out[:, :, crop_h:crop_h+target_h, crop_w:crop_w+target_w]
        return out

class SmallConvNet(nn.Module):
    def __init__(self, in_channels: int, feat_dim: int, nl: Callable[[torch.Tensor], torch.Tensor], last_nl: Optional[Callable[[torch.Tensor], torch.Tensor]], layernormalize: bool):
        super().__init__() # type:ignore
        self.in_channels = in_channels
        self.feat_dim = feat_dim
        self.nl = nl
        self.last_nl = last_nl
        self.layernormalize = layernormalize

        self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc = nn.Linear(3136, feat_dim)
        
        self.apply(init_weights_conv)
        init_weights_fc(self.fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_dim = 1 if x.dim() == 4 else 2
        assert x.shape[channel_dim] == self.in_channels, f"Input channel mismatch: expected {self.in_channels}, got {x.shape[channel_dim]}"
        x = self.nl(self.conv1(x))
        x = self.nl(self.conv2(x))
        x = self.nl(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        if self.last_nl is not None:
            x = self.last_nl(x)
        if self.layernormalize:
            x = layernorm(x)
        return x

class SmallDeconvNet(nn.Module):
    def __init__(self, feat_dim: int, nl: Callable[[torch.Tensor], torch.Tensor], ch: int, positional_bias: bool):
        super().__init__() # type:ignore
        self.nl = nl
        self.ch = ch
        self.positional_bias = positional_bias
        
        self.fc = nn.Linear(feat_dim, 8*8*64)
        init_weights_fc(self.fc)
        
        self.deconv1 = ConvTranspose2dSame(64, 128, kernel_size=4, stride=2)
        self.deconv2 = ConvTranspose2dSame(128, 64, kernel_size=8, stride=2)
        self.deconv3 = ConvTranspose2dSame(64, ch, kernel_size=8, stride=3)
        
        if self.positional_bias:
            self.pos_bias = nn.Parameter(torch.zeros(1, ch, 84, 84))
        
        self.apply(init_weights_conv)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.nl(self.fc(z))
        z = z.view(-1, 64, 8, 8)
        z = self.nl(self.deconv1(z))
        z = self.nl(self.deconv2(z))
        z = self.deconv3(z)
        z = z[:, :, 6:-6, 6:-6]
        if self.positional_bias:
            z = z + self.pos_bias
        return z

class UNet(nn.Module):
    def __init__(self, in_channels: int, feat_dim: int, nl: Callable[[torch.Tensor], torch.Tensor], ac_dim: int):
        super().__init__() # type:ignore
        self.in_channels = in_channels
        self.feat_dim = feat_dim
        self.nl = nl
        self.ac_dim = ac_dim

        c = in_channels + ac_dim
        
        # Encoder
        self.enc1 = Conv2dSame(c, 32, 8, stride=3)
        self.enc2 = Conv2dSame(32 + ac_dim, 64, 8, stride=2)
        self.enc3 = Conv2dSame(64 + ac_dim, 64, 4, stride=2)
        
        self.flat_dim = 8 * 8 * (64 + ac_dim)
        self.fc_in = nn.Linear(self.flat_dim, self.feat_dim)
        
        # Residual Blocks
        self.res_fc1 = nn.ModuleList([nn.Linear(self.feat_dim + ac_dim, self.feat_dim) for _ in range(4)])
        self.res_fc2 = nn.ModuleList([nn.Linear(self.feat_dim + ac_dim, self.feat_dim) for _ in range(4)])
        
        self.fc_out = nn.Linear(self.feat_dim + ac_dim, 8*8*64)
        
        # Decoder (using ADDITION for skip connections, so channel counts match the ADDED output)
        # Skip 1: z_out (64) + l3 (64) = 64. Dec1 input = 64 + ac_dim.
        self.dec1 = ConvTranspose2dSame(64 + ac_dim, 64, 4, stride=2)
        
        # Skip 2: d1 (64) + l2 (64) = 64. Dec2 input = 64 + ac_dim.
        self.dec2 = ConvTranspose2dSame(64 + ac_dim, 32, 8, stride=2)
        
        # Skip 3: d2 (32) + l1 (32) = 32. Dec3 input = 32 + ac_dim.
        self.dec3 = ConvTranspose2dSame(32 + ac_dim, 4, 8, stride=3)
        
        self.apply(init_weights_conv)
        self.apply(init_weights_fc)

    def forward(self, x: torch.Tensor, ac_one_hot: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.in_channels, f"Input channel mismatch: expected {self.in_channels}, got {x.shape[1]}"
        
        def cond(feat: torch.Tensor) -> torch.Tensor:
            B = feat.shape[0]
            if feat.dim() == 4:
                H, W = feat.shape[2], feat.shape[3]
                ac_plane = ac_one_hot.view(B, -1, 1, 1).expand(B, -1, H, W)
                return torch.cat([feat, ac_plane], dim=1)
            else:
                return torch.cat([feat, ac_one_hot], dim=1)

        x_pad = F.pad(x, (6, 6, 6, 6))
        x_in = cond(x_pad)
        
        # Encoder
        l1 = self.nl(self.enc1(x_in))
        l2 = self.nl(self.enc2(cond(l1)))
        l3 = self.nl(self.enc3(cond(l2)))
        
        # Bottleneck
        flat = cond(l3).reshape(l3.size(0), -1)
        z = self.nl(self.fc_in(flat))
        
        for i in range(4):
            res = self.nl(self.res_fc1[i](cond(z)))
            res = self.res_fc2[i](cond(res))
            z = z + res
            
        z_out = self.nl(self.fc_out(cond(z)))
        z_out = z_out.view(-1, 64, 8, 8)
        
        # Decoder with Additive Skip Connections
        z_out = z_out + l3
        d1 = self.nl(self.dec1(cond(z_out)))
        
        d1 = d1 + l2
        d2 = self.nl(self.dec2(cond(d1)))
        
        d2 = d2 + l1
        out = self.dec3(cond(d2))
        
        out = out[:, :, 6:-6, 6:-6]
        return out

def tile_images(array: np.ndarray[Any, Any], n_cols: Optional[int] = None, max_images: Optional[int] = None, div: int = 1) -> np.ndarray[Any, Any]:
    if max_images is not None:
        array = array[:max_images]
    if len(array.shape) == 4 and array.shape[3] == 1:
        array = array[:, :, :, 0]
    assert len(array.shape) in [3, 4], "wrong number of dimensions - shape {}".format(array.shape)
    if len(array.shape) == 4:
        assert array.shape[3] == 3, "wrong number of channels- shape {}".format(array.shape)
    if n_cols is None:
        n_cols = max(int(np.sqrt(array.shape[0])) // div * div, div)
    n_rows = int(np.ceil(float(array.shape[0]) / n_cols))

    def cell(i: int, j: int) -> np.ndarray[Any, Any]:
        ind = i * n_cols + j
        return array[ind] if ind < array.shape[0] else np.zeros(array[0].shape)

    def row(i: int) -> np.ndarray[Any, Any]:
        return np.concatenate([cell(i, j) for j in range(n_cols)], axis=1)

    return np.concatenate([row(i) for i in range(n_rows)], axis=0)