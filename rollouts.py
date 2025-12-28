from collections import deque, defaultdict
import numpy as np
import torch
from typing import List, Tuple, Optional, Union, Any, Dict
from recorder import Recorder
from cnn_policy import CnnPolicy
from dynamics import Dynamics

class Rollout(object):
    def __init__(self, ob_space: Any, ac_space: Any, nenvs: int, nsteps_per_seg: int, nsegs_per_env: int, envs: Any, policy: CnnPolicy,
                 int_rew_coeff: float, ext_rew_coeff: float, record_rollouts: bool, dynamics: Dynamics):
        self.nenvs = nenvs
        self.nsteps_per_seg = nsteps_per_seg
        self.nsegs_per_env = nsegs_per_env
        self.nsteps = self.nsteps_per_seg * self.nsegs_per_env
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.envs = envs
        self.policy = policy
        self.dynamics = dynamics
        self.device = policy.ob_mean.device

        self.int_rew_coeff = int_rew_coeff
        self.ext_rew_coeff = ext_rew_coeff

        self.reward_fun = lambda ext_rew, int_rew: self.ext_rew_coeff * torch.clamp(ext_rew, -1., 1.) + self.int_rew_coeff * int_rew

        # Initialize buffers as Tensors
        # Obs: (nenvs, nsteps, C, H, W) -> NCHW format for PyTorch
        c, h, w = ob_space.shape[2], ob_space.shape[0], ob_space.shape[1] # Gym is HWC
        
        self.buf_obs = torch.zeros((nenvs, self.nsteps, c, h, w), dtype=torch.float32, device=self.device)
        self.buf_obs_last = torch.zeros((nenvs, self.nsegs_per_env, c, h, w), dtype=torch.float32, device=self.device)
        
        # Actions: Assuming Discrete (nenvs, nsteps)
        self.buf_acs = torch.zeros((nenvs, self.nsteps), dtype=torch.long, device=self.device)
        
        self.buf_vpreds = torch.zeros((nenvs, self.nsteps), dtype=torch.float32, device=self.device)
        self.buf_nlps = torch.zeros((nenvs, self.nsteps), dtype=torch.float32, device=self.device)
        self.buf_rews = torch.zeros((nenvs, self.nsteps), dtype=torch.float32, device=self.device)
        self.buf_ext_rews = torch.zeros((nenvs, self.nsteps), dtype=torch.float32, device=self.device)
        self.buf_news = torch.zeros((nenvs, self.nsteps), dtype=torch.float32, device=self.device)
        
        # Last values for GAE bootstrapping
        self.buf_new_last = torch.zeros((nenvs,), dtype=torch.float32, device=self.device)
        self.buf_vpred_last = torch.zeros((nenvs,), dtype=torch.float32, device=self.device)

        self.recorder = Recorder(nenvs=self.nenvs) if record_rollouts else None
        self.statlists: Dict[str, deque] = defaultdict(lambda: deque([], maxlen=100))
        self.stats: Dict[str, Any] = defaultdict(float)
        self.best_ext_ret: Optional[float] = None
        self.all_visited_rooms: List[Any] = []
        self.all_scores: List[float] = []
        self.current_max: Optional[float] = None

        self.step_count = 0
        self.t_last_update = 0

    def collect_rollout(self) -> None:
        self.ep_infos_new: List[Tuple[int, Dict[str, Any]]] = []
        
        # Reset buffers that accumulate per rollout if needed, 
        # but mostly they are overwritten by index `t`.
        
        for t in range(self.nsteps):
            self.rollout_step()
            
        self.calculate_reward()
        self.update_info()

    def calculate_reward(self) -> None:
        # Calculate Intrinsic Reward
        # We process in chunks to avoid OOM, similar to original dynamics.calculate_loss
        
        # buf_obs: (N, T, C, H, W)
        # buf_obs_last: (N, 1, C, H, W) (assuming 1 seg) -> In original loop, updated at end of seg.
        # We need to construct the sequence "next_ob" for the dynamics loss.
        # Dynamics loss expects (obs, last_obs, acs).
        
        n_chunks = 8
        n = self.nenvs
        if n < n_chunks:
            chunk_size = n
            n_chunks = 1
        else:
            chunk_size = n // n_chunks
        
        int_rews = []
        
        with torch.no_grad():
            for i in range(n_chunks):
                # Handle potential uneven split if nenvs not divisible by 8 (though code asserted it)
                start = i * chunk_size
                end = min((i + 1) * chunk_size, n)
                if i == n_chunks - 1:
                    end = n # Ensure we cover everything
                
                if start >= n: break
                
                obs_chunk = self.buf_obs[start:end]
                last_obs_chunk = self.buf_obs_last[start:end] # (B, 1, C, H, W)
                acs_chunk = self.buf_acs[start:end]
                
                # dynamics.get_loss returns (B, T)
                loss = self.dynamics.get_loss(obs_chunk, last_obs_chunk, acs_chunk)
                int_rews.append(loss)
                
        int_rew = torch.cat(int_rews, dim=0) # (N, T)
        
        # Total Reward
        self.buf_rews = self.reward_fun(ext_rew=self.buf_ext_rews, int_rew=int_rew)

    def rollout_step(self) -> None:
        t = self.step_count % self.nsteps
        s = t % self.nsteps_per_seg
        
        # 1. Get Action from Policy
        obs, prevrews, news, infos = self.env_get()
        
        # Update infos
        for info in infos:
            epinfo = info.get('episode', {})
            mzepinfo = info.get('mz_episode', {})
            retroepinfo = info.get('retro_episode', {})
            epinfo.update(mzepinfo)
            epinfo.update(retroepinfo)
            if epinfo:
                if "n_states_visited" in info:
                    epinfo["n_states_visited"] = info["n_states_visited"]
                    epinfo["states_visited"] = info["states_visited"]
                self.ep_infos_new.append((self.step_count, epinfo))

        # Prepare Obs for Policy
        # obs is Numpy (N, H, W, C). Convert to Tensor (N, C, H, W) on device.
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        obs_tensor = obs_tensor.permute(0, 3, 1, 2)
        
        # Add time dim for policy: (N, 1, C, H, W)
        obs_input = obs_tensor.unsqueeze(1)
        
        with torch.no_grad():
            # a_samp: (N, 1), vpred: (N, 1), nlp: (N, 1)
            a_samp, vpred, nlp, _, _ = self.policy.forward(obs_input)
        
        # Squeeze time dim
        acs = a_samp.squeeze(1)
        vpreds = vpred.squeeze(1)
        nlps = nlp.squeeze(1)
        
        # 2. Step Env
        # Convert actions to numpy for Gym
        acs_np = acs.cpu().numpy()
        self.env_step(acs_np)
        
        # 3. Store Data
        # Store obs_tensor.
        self.buf_obs[:, t] = obs_tensor
        self.buf_news[:, t] = torch.from_numpy(news).float().to(self.device)
        self.buf_vpreds[:, t] = vpreds
        self.buf_nlps[:, t] = nlps
        self.buf_acs[:, t] = acs
        
        # Store prev rewards (ext)
        if t > 0:
            self.buf_ext_rews[:, t - 1] = torch.from_numpy(prevrews).float().to(self.device)

        # Recorder (uses numpy)
        if self.recorder is not None:
            
            # Let's keep a dummy int_rew buffer for recorder if needed, or pass 0.
            current_int_rew = np.zeros(self.nenvs, dtype=np.float32) # BIG TODO: fix
            self.recorder.record(timestep=self.step_count, acs=acs_np, infos=infos, 
                                 int_rew=current_int_rew,
                                 ext_rew=prevrews, news=news)

        self.step_count += 1
        
        # End of Segment/Rollout logic
        if s == self.nsteps_per_seg - 1:
            nextobs, ext_rews, nextnews, _ = self.env_get()
            
            # Convert
            nextobs_tensor = torch.from_numpy(nextobs).float().to(self.device).permute(0, 3, 1, 2)
            
            # Store 'next_obs' in buf_obs_last
            # t // nsteps_per_seg gives segment index
            self.buf_obs_last[:, t // self.nsteps_per_seg] = nextobs_tensor
            
            if t == self.nsteps - 1:
                self.buf_new_last[:] = torch.from_numpy(nextnews).float().to(self.device)
                self.buf_ext_rews[:, t] = torch.from_numpy(ext_rews).float().to(self.device)
                
                # Value of last obs
                obs_input = nextobs_tensor.unsqueeze(1)
                with torch.no_grad():
                     _, vpred_last, _, _, _ = self.policy.forward(obs_input)
                self.buf_vpred_last[:] = vpred_last.squeeze(1)

    def update_info(self) -> None:
        # Single GPU, no gathering needed
        all_ep_infos = self.ep_infos_new
        all_ep_infos = sorted(all_ep_infos, key=lambda x: x[0])
        
        if all_ep_infos:
            all_ep_infos_vals = [i_[1] for i_ in all_ep_infos]
            keys_ = all_ep_infos_vals[0].keys()
            all_ep_infos_dict = {k: [i[k] for i in all_ep_infos_vals] for k in keys_}

            self.statlists['eprew'].extend(all_ep_infos_dict['r'])
            self.stats['eprew_recent'] = np.mean(all_ep_infos_dict['r'])
            self.statlists['eplen'].extend(all_ep_infos_dict['l'])
            self.stats['epcount'] += len(all_ep_infos_dict['l'])
            self.stats['tcount'] += sum(all_ep_infos_dict['l'])
            
            if 'visited_rooms' in keys_:
                self.stats['visited_rooms'] = sorted(list(set.union(*all_ep_infos_dict['visited_rooms'])))
                self.stats['pos_count'] = np.mean(all_ep_infos_dict['pos_count'])
                self.all_visited_rooms.extend(self.stats['visited_rooms'])
                self.all_scores.extend(all_ep_infos_dict["r"])
                self.all_scores = sorted(list(set(self.all_scores)))
                self.all_visited_rooms = sorted(list(set(self.all_visited_rooms)))
                print("All visited rooms")
                print(self.all_visited_rooms)
                print("All scores")
                print(self.all_scores)
                
            if 'levels' in keys_:
                temp = sorted(list(set.union(*all_ep_infos_dict['levels'])))
                self.all_visited_rooms.extend(temp)
                self.all_visited_rooms = sorted(list(set(self.all_visited_rooms)))
                print("All visited levels")
                print(self.all_visited_rooms)

            current_max = np.max(all_ep_infos_dict['r'])
        else:
            current_max = None
        
        self.ep_infos_new = []

        if current_max is not None:
            if (self.best_ext_ret is None) or (current_max > self.best_ext_ret):
                self.best_ext_ret = current_max
        self.current_max = current_max

    # Wrapper methods for the single VecEnv to match original structure
    def env_step(self, acs: np.ndarray[Any, Any]) -> None:
        self.envs.step_async(acs)
        self.env_result = None

    def env_get(self) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], List[Dict[str, Any]]]:
        # In step 0 or if explicitly called:
        if self.step_count == 0:
            ob = self.envs.reset()
            # Construct standard tuple: (obs, rews, dones, infos)
            # Reset only returns obs. 
            # We create dummy rews/dones/infos
            n = self.nenvs
            out = (ob, np.zeros(n, dtype=np.float32), np.zeros(n, dtype=bool), [{} for _ in range(n)])
            self.env_result = out
        else:
            if self.env_result is None:
                self.env_result = self.envs.step_wait()
            out = self.env_result
        return out