import os
import pickle
from typing import List, Tuple, Optional, Union, Any, Dict
from baselines import logger

class Recorder(object):
    def __init__(self, nenvs: int, nlumps: int):
        # nlumps is deprecated/unused in single-process version but kept for signature compatibility
        self.nenvs = nenvs
        self.acs: List[List[Any]] = [[] for _ in range(nenvs)]
        self.int_rews: List[List[float]] = [[] for _ in range(nenvs)]
        self.ext_rews: List[List[float]] = [[] for _ in range(nenvs)]
        self.ep_infos: List[Dict[str, Any]] = [{} for _ in range(nenvs)]
        self.filenames = [self.get_filename(i) for i in range(nenvs)]
        
        # Always log since we are effectively Rank 0
        logger.info("episode recordings saved to ", self.filenames[0])

    def record(self, timestep: int, lump: int, acs: Any, infos: List[Dict[str, Any]], int_rew: Any, ext_rew: Any, news: Any) -> None:
        # lump argument is ignored
        # acs, int_rew, ext_rew, news are expected to be arrays of length nenvs
        
        for i in range(self.nenvs):
            if timestep == 0:
                self.acs[i].append(acs[i])
            else:
                if self.is_first_episode_step(i):
                    try:
                        self.ep_infos[i]['random_state'] = infos[i]['random_state']
                    except:
                        pass

                self.int_rews[i].append(int_rew[i])
                self.ext_rews[i].append(ext_rew[i])

                if news[i]:
                    # Extract episode info if available
                    # infos[i] might contain 'episode' key from Monitor wrapper
                    if 'episode' in infos[i]:
                        self.ep_infos[i]['ret'] = infos[i]['episode']['r']
                        self.ep_infos[i]['len'] = infos[i]['episode']['l']
                        self.dump_episode(i)

                self.acs[i].append(acs[i])

    def dump_episode(self, i: int) -> None:
        episode = {'acs': self.acs[i],
                   'int_rew': self.int_rews[i],
                   'info': self.ep_infos[i]}
        filename = self.filenames[i]
        
        # Only save env 0 to save disk space, matching original logic
        if self.episode_worth_saving(i):
            with open(filename, 'ab') as f:
                pickle.dump(episode, f, protocol=-1)
        
        self.acs[i].clear()
        self.int_rews[i].clear()
        self.ext_rews[i].clear()
        self.ep_infos[i].clear()

    def episode_worth_saving(self, i: int) -> bool:
        # Original logic: return (i == 0 and MPI.COMM_WORLD.Get_rank() == 0)
        # We are always rank 0.
        return i == 0

    def is_first_episode_step(self, i: int) -> bool:
        return len(self.int_rews[i]) == 0

    def get_filename(self, i: int) -> str:
        # Replaced MPI rank with 0
        filename = os.path.join(logger.get_dir(), 'env{}_{}.pk'.format(0, i))
        return filename