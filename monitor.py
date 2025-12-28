import gym
import time
import csv
import os
import json
from typing import Tuple, Dict, Any, Optional

class Monitor(gym.Wrapper[Any, Any]):
    def __init__(self, env: gym.Env[Any, Any], filename: Optional[str] = None, allow_early_resets: bool = False, reset_keywords=(), info_keywords=()):
        super().__init__(env)
        self.tstart = time.time()
        if filename:
            self.results_writer = ResultsWriter(filename, header={'t_start': self.tstart, 'env_id': env.spec.id if env.spec else 'unknown'},
                                                extra_keys=reset_keywords + info_keywords)
        else:
            self.results_writer = None
            
        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = []
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {} # extra info about the current episode, that was passed in during reset()

    def reset(self, **kwargs):
        self.reset_state()
        return self.env.reset(**kwargs)

    def reset_state(self):
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError("Tried to reset an environment before done. If you want to allow early resets, wrap your env with Monitor(env, path, allow_early_resets=True)")
        self.rewards = []
        self.needs_reset = False
        self.t_start = time.time()

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        
        result = self.env.step(action)
        
        # Handle new gym API (obs, rew, term, trunc, info) or (obs, rew, done, info)
        if len(result) == 5:
            obs, rew, term, trunc, info = result
            done = term or trunc
        elif len(result) == 4:
            obs, rew, done, info = result
        else:
            raise ValueError("Env returned unexpected number of values")

        self.rewards.append(rew)
        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            for k in self.info_keywords:
                ep_info[k] = info[k]
            self.episode_rewards.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            
            if isinstance(info, dict):
                info['episode'] = ep_info
        
        self.total_steps += 1
        
        if len(result) == 5:
            return obs, rew, term, trunc, info
        else:
            return obs, rew, done, info

class ResultsWriter:
    def __init__(self, filename, header=None, extra_keys=()):
        self.extra_keys = extra_keys
        if filename is not None:
            if filename.endswith('.csv'):
                filename = filename[:-4]
            self.f = open(filename + '.monitor.csv', "wt")
            if header:
                self.f.write('#%s\n' % json.dumps(header))
            self.logger = csv.DictWriter(self.f, fieldnames=('r', 'l', 't') + tuple(extra_keys))
            self.logger.writeheader()
            self.f.flush()
        else:
            self.f = None

    def write_row(self, epinfo):
        if self.logger:
            self.logger.writerow(epinfo)
            self.f.flush()
