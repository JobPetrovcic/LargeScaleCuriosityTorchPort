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
        if self.normrew:
            # TF: rffs = np.array([self.rff.update(rew) for rew in self.rollout.buf_rews.T])
            orig_rews = self.rollout.buf_rews.cpu().numpy() # (N, T)
            rffs_list = []
            for t in range(orig_rews.shape[1]):
                rffs_list.append(self.rff.update(orig_rews[:, t]))
            rffs = np.array(rffs_list).T # (N, T)
            
            rffs_mean, rffs_std = get_mean_and_std(rffs.ravel())
            rffs_count = len(rffs.ravel())
            
            rffs_mean_t = torch.tensor(rffs_mean, device=self.device, dtype=torch.float64)
            rffs_var_t = torch.tensor(rffs_std ** 2, device=self.device, dtype=torch.float64)
            
            self.rff_rms.update_from_moments(rffs_mean_t, rffs_var_t, rffs_count)
            
            rews = self.rollout.buf_rews / (torch.sqrt(self.rff_rms.var.float()) + 1e-8)
        else:
            rews = self.rollout.buf_rews
        
        self.calculate_advantages(rews=rews, use_news=self.use_news, gamma=self.gamma, lam=self.lam)

        info = dict(
            advmean=self.buf_advs.mean().item(),
            advstd=self.buf_advs.std().item(),
            retmean=self.buf_rets.mean().item(),
            retstd=self.buf_rets.std().item(),
            vpredmean=self.rollout.buf_vpreds.mean().item(),
            vpredstd=self.rollout.buf_vpreds.std().item(),
            ev=1.0 - (torch.var(self.buf_rets - self.rollout.buf_vpreds) / (torch.var(self.buf_rets) + 1e-8)).item(),
            rew_mean=torch.mean(self.rollout.buf_rews).item(),
            recent_best_ext_ret=self.rollout.current_max
        )
        if self.rollout.best_ext_ret is not None:
            info['best_ext_ret'] = self.rollout.best_ext_ret

        # normalize advantages
        if self.normadv:
            m = self.buf_advs.mean()
            s = self.buf_advs.std()
            self.buf_advs = (self.buf_advs - m) / (s + 1e-7)

        envsperbatch = (self.nenvs * self.nsegs_per_env) // self.nminibatches
        envsperbatch = max(1, envsperbatch)
        envinds = np.arange(self.nenvs * self.nsegs_per_env)

        def resh(x):
            if self.nsegs_per_env == 1:
                return x
            # x: (N, T, ...)
            sh = x.shape
            # Reshape to (N*S, T/S, ...)
            new_sh = (sh[0], self.nsegs_per_env, self.nsteps_per_seg) + sh[2:]
            return x.reshape(new_sh).flatten(0, 1)

        # Prepare buffers
        b_obs = resh(self.rollout.buf_obs)
        b_acs = resh(self.rollout.buf_acs)
        b_vpreds = resh(self.rollout.buf_vpreds)
        b_nlps = resh(self.rollout.buf_nlps)
        b_rets = resh(self.buf_rets)
        b_advs = resh(self.buf_advs)
        
        # Last obs: (N, S, ...) -> (N*S, 1, ...)
        # buf_obs_last is (N, S, C, H, W). We want (N*S, 1, C, H, W).
        c, h, w = self.ob_space.shape[2], self.ob_space.shape[0], self.ob_space.shape[1]
        b_last_obs = self.rollout.buf_obs_last.reshape(self.nenvs * self.nsegs_per_env, 1, c, h, w)
        
        mblossvals = []

        for _ in range(self.nepochs):
            np.random.shuffle(envinds)
            for start in range(0, self.nenvs * self.nsegs_per_env, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                
                mb_obs = b_obs[mbenvinds]
                mb_acs = b_acs[mbenvinds]
                mb_vpreds = b_vpreds[mbenvinds]
                mb_nlps = b_nlps[mbenvinds]
                mb_rets = b_rets[mbenvinds]
                mb_advs = b_advs[mbenvinds]
                mb_last_obs = b_last_obs[mbenvinds]
                
                # Forward
                _, vpred, nlp, entropy, _ = self.stochpol.forward(mb_obs, mb_acs)
                
                # Losses
                ent_loss = -self.ent_coeff * entropy.mean()
                vf_loss = 0.5 * torch.mean((vpred - mb_rets) ** 2)
                
                ratio = torch.exp(mb_nlps - nlp)
                pg_losses1 = -mb_advs * ratio
                pg_losses2 = -mb_advs * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
                pg_loss = torch.max(pg_losses1, pg_losses2).mean()
                
                with torch.no_grad():
                    approxkl = 0.5 * torch.mean((nlp - mb_nlps) ** 2)
                    clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.cliprange).float())
                
                dyn_loss = self.dynamics.get_loss(mb_obs, mb_last_obs, mb_acs)
                dyn_loss_mean = dyn_loss.mean()
                
                aux_loss = self.dynamics.auxiliary_task.get_loss(mb_obs, mb_last_obs, mb_acs)
                if aux_loss.numel() > 1:
                    aux_loss = aux_loss.mean()
                    
                with torch.no_grad():
                    feats = self.dynamics.auxiliary_task.get_features(mb_obs)
                    feat_var = feats.var(dim=[0, 1], unbiased=False).mean()
                    
                loss = pg_loss + vf_loss + ent_loss + dyn_loss_mean + aux_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                mblossvals.append([loss.item(), pg_loss.item(), vf_loss.item(), ent_loss.item(), 
                                  approxkl.item(), clipfrac.item(), aux_loss.item(), dyn_loss_mean.item(), feat_var.item()])

        # Report
        loss_names_full = ['tot', 'pg', 'vf', 'ent', 'approxkl', 'clipfrac', 'aux', 'dyn_loss', 'feat_var']
        mean_losses = np.mean(mblossvals, axis=0)
        
        for i, name in enumerate(loss_names_full):
            info['opt_' + name] = mean_losses[i]

        self.n_updates += 1
        info["n_updates"] = self.n_updates
        
        for dn, dvs in self.rollout.statlists.items():
            info[dn] = np.mean(dvs) if len(dvs) > 0 else 0
        info.update(self.rollout.stats)
        if "states_visited" in info:
            info.pop("states_visited")
            
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