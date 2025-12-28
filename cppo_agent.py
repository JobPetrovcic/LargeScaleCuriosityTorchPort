import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import BatchSampler, SubsetRandomSampler
from typing import Any, List, Tuple, Optional, Dict, Union

from rollouts import Rollout
from utils import RunningMeanStd, get_mean_and_std
from dynamics import Dynamics
from cnn_policy import CnnPolicy

class RewardForwardFilter(object):
    def __init__(self, gamma: float):
        self.rewems: Optional[Union[np.ndarray[Any, Any], torch.Tensor]] = None
        self.gamma = gamma

    def update(self, rews: Union[np.ndarray[Any, Any], torch.Tensor]) -> Union[np.ndarray[Any, Any], torch.Tensor]:
        # rews: numpy array or tensor? 
        # rollouts.py buf_rews is Tensor.
        # But this filter runs on CPU usually? 
        # The original code did: "rffs = np.array([self.rff.update(rew) for rew in self.rollout.buf_rews.T])"
        # It iterated over time steps (columns).
        
        # We'll implement a vectorized version if possible, or stick to loop for exactness.
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems

class PpoOptimizer(object):
    def __init__(self, scope: str, ob_space: Any, ac_space: Any, stochpol: CnnPolicy,
                 ent_coef: float, gamma: float, lam: float, nepochs: int, lr: float, cliprange: float,
                 nminibatches: int,
                 normrew: bool, normadv: bool, use_news: bool, ext_coeff: float, int_coeff: float,
                 nsteps_per_seg: int, nsegs_per_env: int, dynamics: Dynamics):
        self.scope = scope
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.stochpol = stochpol
        self.nepochs = nepochs
        self.lr = lr
        self.cliprange = cliprange
        self.nsteps_per_seg = nsteps_per_seg
        self.nsegs_per_env = nsegs_per_env
        self.nminibatches = nminibatches
        self.gamma = gamma
        self.lam = lam
        self.normrew = normrew
        self.normadv = normadv
        self.use_news = use_news
        self.ext_coeff = ext_coeff
        self.int_coeff = int_coeff
        self.ent_coeff = ent_coef
        self.dynamics = dynamics
        self.device = stochpol.ob_mean.device
        
        self.use_recorder = True
        self.n_updates = 0
        
        # Optimizer
        # TF Code: AdamOptimizer(learning_rate=ph_lr)
        # We need to collect parameters from policy, dynamics, and auxiliary tasks.
        # auxiliary_tasks are submodules of dynamics (usually).
        # We need to ensure we get all trainable params.
        
        self.params = list(self.stochpol.parameters()) + list(self.dynamics.parameters())
        # remove duplicates
        self.params = list({id(p): p for p in self.params}.values())
        self.optimizer = optim.Adam(self.params, lr=self.lr, eps=1e-8)
        
        self.loss_names = ['tot', 'pg', 'vf', 'ent', 'approxkl', 'clipfrac', 'aux', 'dyn_loss', 'feat_var']
        self.to_report: Dict[str, float] = {} # Placeholder

    def start_interaction(
            self, 
            envs: Any, 
            dynamics: Dynamics # BIG TODO: why is this unused?
        ) -> None:
        self.nenvs = envs.num_envs
        self.envs = envs
        
        # Initialize Rollout storage
        self.rollout = Rollout(ob_space=self.ob_space, ac_space=self.ac_space, nenvs=self.nenvs,
                               nsteps_per_seg=self.nsteps_per_seg,
                               nsegs_per_env=self.nsegs_per_env,
                               envs=self.envs,
                               policy=self.stochpol,
                               int_rew_coeff=self.int_coeff,
                               ext_rew_coeff=self.ext_coeff,
                               record_rollouts=self.use_recorder,
                               dynamics=self.dynamics)

        # Buffers for Advantage Calculation
        self.buf_advs = torch.zeros((self.nenvs, self.rollout.nsteps), dtype=torch.float32, device=self.device)
        self.buf_rets = torch.zeros((self.nenvs, self.rollout.nsteps), dtype=torch.float32, device=self.device)

        if self.normrew:
            self.rff = RewardForwardFilter(self.gamma)
            self.rff_rms = RunningMeanStd(shape=(), epsilon=1e-4)
            # self.rff_rms needs to be on same device? It's a module.
            self.rff_rms.to(self.device)

        self.step_count = 0
        self.t_last_update = time.time()
        self.t_start = time.time()

    def stop_interaction(self) -> None:
        self.envs.close()

    def calculate_advantages(self, rews: torch.Tensor, use_news: bool, gamma: float, lam: float) -> None:
        # rews: Tensor (N, T)
        nsteps = self.rollout.nsteps
        lastgaelam = 0
        
        # We need access to vpreds and news from rollout
        # buf_vpreds: (N, T)
        # buf_news: (N, T)
        
        # To make this efficient on GPU, we can't easily loop backwards T times with python scalar logic 
        # if we want max speed, but for nsteps=128 it's fine.
        
        vpreds = self.rollout.buf_vpreds
        news = self.rollout.buf_news
        vpred_last = self.rollout.buf_vpred_last
        new_last = self.rollout.buf_new_last
        
        # Ensure target buffer is clean
        self.buf_advs.zero_()
        
        for t in range(nsteps - 1, -1, -1):
            if t + 1 < nsteps:
                nextnew = news[:, t + 1]
                nextvals = vpreds[:, t + 1]
            else:
                nextnew = new_last
                nextvals = vpred_last
            
            if not use_news:
                nextnew = torch.zeros_like(nextnew)
            
            nextnotnew = 1.0 - nextnew
            delta = rews[:, t] + gamma * nextvals * nextnotnew - vpreds[:, t]
            
            lastgaelam = delta + gamma * lam * nextnotnew * lastgaelam
            self.buf_advs[:, t] = lastgaelam
            
        self.buf_rets[:] = self.buf_advs + vpreds

    def update(self) -> Dict[str, float]:
        # 1. Normalize Rewards (if enabled)
        if self.normrew:
            # TF: "rffs = np.array([self.rff.update(rew) for rew in self.rollout.buf_rews.T])"
            # Iterates columns (timesteps). buf_rews is (N, T).
            
            # We move to CPU for RFF update to match original scalar logic if needed, 
            # or do it on GPU. Since RFF is recursive state, iterating columns is necessary.
            
            # Note: The original code does `buf_rews.T` (T, N).
            # self.rff.update takes `rews` (N,).
            
            orig_rews = self.rollout.buf_rews.cpu().numpy() # (N, T)
            rffs_list = []
            for t in range(orig_rews.shape[1]):
                rffs_list.append(self.rff.update(orig_rews[:, t]))
            
            rffs = np.array(rffs_list).T # (N, T) -- undo the T from loop
            
            # Update RMS
            rffs_mean, rffs_std = get_mean_and_std(rffs.ravel())
            
            # RunningMeanStd update needs batch_mean, batch_var, count.
            # We can use the helper directly if we convert to torch.
            rffs_tensor = torch.from_numpy(rffs).to(self.device).double()
            self.rff_rms.update(rffs_tensor.reshape(-1))
            
            # Normalize rews
            # "rews = self.rollout.buf_rews / np.sqrt(self.rff_rms.var)"
            rews = self.rollout.buf_rews / (torch.sqrt(self.rff_rms.var.float()) + 1e-8) # Safety eps
        else:
            rews = self.rollout.buf_rews
        
        # 2. Calculate Advantages
        self.calculate_advantages(rews=rews, use_news=self.use_news, gamma=self.gamma, lam=self.lam)

        # 3. Info Stats
        # We use .item() to get scalars
        info = dict(
            advmean=self.buf_advs.mean().item(),
            advstd=self.buf_advs.std().item(),
            retmean=self.buf_rets.mean().item(),
            retstd=self.buf_rets.std().item(),
            vpredmean=self.rollout.buf_vpreds.mean().item(),
            vpredstd=self.rollout.buf_vpreds.std().item(),
            # ev = 1 - Var(y-pred)/Var(y)
            ev=1.0 - (torch.var(self.buf_rets - self.rollout.buf_vpreds) / (torch.var(self.buf_rets) + 1e-8)).item(),
            rew_mean=torch.mean(self.rollout.buf_rews).item(),
            recent_best_ext_ret=self.rollout.current_max
        )
        if self.rollout.best_ext_ret is not None:
            info['best_ext_ret'] = self.rollout.best_ext_ret

        # 4. Normalize Advantages
        if self.normadv:
            mean = self.buf_advs.mean()
            std = self.buf_advs.std()
            self.buf_advs = (self.buf_advs - mean) / (std + 1e-8) # TF uses 1e-7

        # 5. Prepare Batch
        # Flatten everything: (N * T, ...)
        
        # Helper to reshape (N, T, ...) -> (N*T, ...)
        def flat(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(-1, *x.shape[2:])
            
        b_obs = flat(self.rollout.buf_obs)        # (NT, C, H, W)
        b_acs = flat(self.rollout.buf_acs)        # (NT,)
        b_logprobs = flat(self.rollout.buf_nlps)  # (NT,) - old nlp
        b_returns = flat(self.buf_rets)           # (NT,)
        b_advs = flat(self.buf_advs)              # (NT,)
        b_values = flat(self.rollout.buf_vpreds)  # (NT,) - old vpred
        
        # Last observation (for dynamics loss if needed by logic)
        # Original: passed rollout.buf_obs_last separately.
        # But dynamics loss needs (obs, last_obs). 
        # `rollout.buf_obs` covers 0...T-1. `buf_obs_last` covers T.
        # WAIT. In `dynamics.get_loss(obs, last_obs)`, `last_obs` is usually the *next* observation for the last step.
        # But in PPO update, we don't usually re-calculate dynamics loss unless we are updating dynamics.
        # Yes, we are updating dynamics.
        # TF Code: "ph_buf.extend([(self.dynamics.last_ob, rollout.buf_obs_last...)])"
        # We need to construct the dataset such that for every obs[t], we have obs[t+1].
        
        # We need (N, T, ...) flattened.
        # obs[t] -> obs[t+1].
        # For t < T-1, obs[t+1] is in buf_obs.
        # For t = T-1, obs[t+1] is in buf_obs_last.
        
        # Let's construct `b_next_obs` (NT, C, H, W).
        # We can construct it before flattening.
        next_obs = torch.cat([self.rollout.buf_obs[:, 1:], self.rollout.buf_obs_last], dim=1) # (N, T, C, H, W)
        b_next_obs = flat(next_obs)
        
        total_samples = b_obs.shape[0]
        batch_size = total_samples // self.nminibatches
        
        indices = np.arange(total_samples)
        
        loss_vals = []
        
        for epoch in range(self.nepochs):
            np.random.shuffle(indices)
            
            for start in range(0, total_samples, batch_size):
                end = start + batch_size
                mb_inds = indices[start:end]
                
                # Slices
                mb_obs = b_obs[mb_inds]
                mb_acs = b_acs[mb_inds]
                mb_next_obs = b_next_obs[mb_inds]
                mb_old_logprobs = b_logprobs[mb_inds]
                mb_returns = b_returns[mb_inds]
                mb_advs = b_advs[mb_inds]
                mb_old_vals = b_values[mb_inds]
                
                # --- Forward Pass ---
                
                # 1. Policy Loss
                # We need to reshape obs to add dummy time dim? 
                # CnnPolicy.forward takes (B, T, ...). Here we have (B, ...).
                # We can treat batch as T=1, or just unsqueeze.
                # The policy code flattens internally anyway.
                
                # Add time dim: (MB, 1, C, H, W)
                mb_obs_t = mb_obs.unsqueeze(1)
                mb_acs_t = mb_acs.unsqueeze(1)
                
                # Get current policy outputs
                _, vpred, nlp, entropy, _ = self.stochpol.forward(mb_obs_t, mb_acs_t)
                # Outputs are (MB, 1). Squeeze.
                vpred = vpred.squeeze(1)
                nlp = nlp.squeeze(1)
                entropy = entropy.mean() # Mean over batch
                
                # Ratio
                ratio = torch.exp(mb_old_logprobs - nlp)
                
                # Policy Loss
                pg_losses1 = -mb_advs * ratio
                pg_losses2 = -mb_advs * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
                pg_loss = torch.max(pg_losses1, pg_losses2).mean()
                
                # Value Loss
                vf_loss = 0.5 * torch.mean((vpred - mb_returns) ** 2)
                
                # Entropy Loss
                ent_loss = -self.ent_coeff * entropy
                
                # Approx KL (for reporting)
                with torch.no_grad():
                    approxkl = 0.5 * torch.mean((nlp - mb_old_logprobs) ** 2)
                    clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.cliprange).float())
                
                # 2. Dynamics / Aux Loss
                # Dynamics needs (obs, next_obs, acs).
                # Note: `next_obs` corresponds to `last_ob` in dynamics.get_loss signature, 
                # but conceptually it's just the next observation.
                # We passed `b_next_obs` as `last_ob` argument.
                # mb_obs: (MB, C, H, W).
                # mb_next_obs: (MB, C, H, W).
                # We need to add time dim for dynamics too?
                # dynamics.get_loss expects (B, T, ...).
                
                mb_next_obs_t = mb_next_obs.unsqueeze(1)
                
                dyn_loss = self.dynamics.get_loss(mb_obs_t, mb_next_obs_t, mb_acs_t)
                dyn_loss_mean = dyn_loss.mean()
                
                # Aux Task Loss
                # "self.feature_extractor.loss" in TF code.
                # FeatureExtractor.get_loss(obs, last_ob, acs)
                # But wait, Aux task loss is often part of Dynamics loss or separate?
                # In TF init: "self.loss = self.get_loss()" inside FeatureExtractor.
                # In Trainer: "self.agent.to_report['aux'] = tf.reduce_mean(self.feature_extractor.loss)"
                # FeatureExtractor (VAE/IDF) loss logic:
                # IDF: predicts action.
                # VAE: reconstruction.
                
                # We need to call aux_task.get_loss() explicitly.
                aux_loss = self.dynamics.auxiliary_task.get_loss(mb_obs_t, mb_next_obs_t, mb_acs_t)
                if aux_loss.numel() > 1: # Reduce if necessary
                    aux_loss = aux_loss.mean()
                
                # Feature variation (for reporting)
                with torch.no_grad():
                    feats = self.dynamics.auxiliary_task.get_features(mb_obs_t)
                    # feats: (B, T, D)
                    # TF: moments(features, [0, 1]). [0,1] is Batch, Time.
                    # We have (MB, 1, D).
                    feat_var = feats.var(dim=[0, 1], unbiased=False).mean()
                
                # Total Loss
                loss = pg_loss + vf_loss + ent_loss + dyn_loss_mean + aux_loss
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                
                # TF MpiAdamOptimizer does global norm clipping? 
                # Original code doesn't explicitly show clipping in PpoOptimizer, 
                # but MpiAdamOptimizer averages grads.
                # Baselines usually clips to 0.5. 
                # The code provided doesn't show `clip_grad_norm`.
                # We'll assume standard Adam behavior.
                self.optimizer.step()
                
                # Record
                loss_vals.append([loss.item(), pg_loss.item(), vf_loss.item(), entropy.item(), 
                                  approxkl.item(), clipfrac.item(), aux_loss.item(), dyn_loss_mean.item(), feat_var.item()])

        # Report
        loss_names_full = ['tot', 'pg', 'vf', 'ent', 'approxkl', 'clipfrac', 'aux', 'dyn_loss', 'feat_var']
        mean_losses = np.mean(loss_vals, axis=0)
        
        for i, name in enumerate(loss_names_full):
            info['opt_' + name] = mean_losses[i]

        self.n_updates += 1
        info["n_updates"] = self.n_updates
        
        # Add Rollout stats
        for dn, dvs in self.rollout.statlists.items():
            info[dn] = np.mean(dvs) if len(dvs) > 0 else 0
        info.update(self.rollout.stats)
        
        # Timing
        tnow = time.time()
        info["ups"] = 1. / (tnow - self.t_last_update)
        info["total_secs"] = tnow - self.t_start
        info['tps'] = self.rollout.nsteps * self.nenvs / (tnow - self.t_last_update)
        self.t_last_update = tnow

        return info

    def step(self):
        self.rollout.collect_rollout()
        update_info = self.update()
        return {'update': update_info}

    def get_var_values(self):
        return self.stochpol.state_dict()

    def set_var_values(self, vv):
        self.stochpol.load_state_dict(vv)