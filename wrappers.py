import itertools
from collections import deque
from copy import copy
from typing import List, Tuple, Optional, Union, Any, Dict, Set

import gym
import numpy as np
from PIL import Image

# Helper to detect gym version for API compatibility
is_modern_gym = hasattr(gym, 'version') and gym.__version__ >= '0.26'

def unwrap(env: gym.Env) -> gym.Env:
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int = 4):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer: deque = deque(maxlen=2)
        self._skip = skip

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        terminated = False
        truncated = False
        acc_info: Dict[str, Any] = {}
        
        for _ in range(self._skip):
            step_result = self.env.step(action)
            # Handle Gym API differences
            if len(step_result) == 5:
                obs, reward, term, trunc, info = step_result
                done = term or trunc
            else:
                obs, reward, done, info = step_result
                term, trunc = done, False

            acc_info.update(info)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                terminated = term
                truncated = trunc
                break
                
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        if is_modern_gym:
            return max_frame, total_reward, terminated, truncated, acc_info
        else:
            return max_frame, total_reward, terminated or truncated, acc_info

    def reset(self, **kwargs: Any) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        
        reset_result = self.env.reset(**kwargs)
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            info = {} # Fallback
            
        self._obs_buffer.append(obs)
        
        if is_modern_gym:
            return obs, info
        return obs

class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, crop: bool = True):
        self.crop = crop
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return ProcessFrame84.process(obs, crop=self.crop)

    @staticmethod
    def process(frame: np.ndarray, crop: bool = True) -> np.ndarray:
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        elif frame.size == 224 * 240 * 3:  # mario resolution
            img = np.reshape(frame, [224, 240, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution." + str(frame.size)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        size = (84, 110 if crop else 84)
        resized_screen = np.array(Image.fromarray(img).resize(size,
                                                              resample=Image.BILINEAR), dtype=np.uint8)
        x_t = resized_screen[18:102, :] if crop else resized_screen
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)

class ExtraTimeLimit(gym.Wrapper):
    def __init__(self, env: gym.Env, max_episode_steps: Optional[int] = None):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        step_result = self.env.step(action)
        if len(step_result) == 5:
            obs, reward, term, trunc, info = step_result
        else:
            obs, reward, done, info = step_result
            term, trunc = done, False
        
        self._elapsed_steps += 1
        if self._max_episode_steps is not None and self._elapsed_steps > self._max_episode_steps:
            trunc = True
            
        if is_modern_gym:
            return obs, reward, term, trunc, info
        else:
            return obs, reward, term or trunc, info

    def reset(self, **kwargs: Any) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class AddRandomStateToInfo(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.random_state_copy: Optional[Any] = None

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        step_result = self.env.step(action)
        if len(step_result) == 5:
            ob, r, term, trunc, info = step_result
        else:
            ob, r, done, info = step_result
            term, trunc = done, False
            
        if self.random_state_copy is not None:
            info['random_state'] = self.random_state_copy
            self.random_state_copy = None
            
        if is_modern_gym:
            return ob, r, term, trunc, info
        else:
            return ob, r, term or trunc, info

    def reset(self, **kwargs: Any) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
        self.random_state_copy = copy(unwrap(self.env).np_random)
        return self.env.reset(**kwargs)

class MontezumaInfoWrapper(gym.Wrapper):
    ram_map = {
        "room": dict(index=3),
        "x": dict(index=42),
        "y": dict(index=43),
    }

    def __init__(self, env: gym.Env):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.visited: Set[Tuple[int, int, int]] = set()
        self.visited_rooms: Set[int] = set()

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        step_result = self.env.step(action)
        if len(step_result) == 5:
            obs, rew, term, trunc, info = step_result
            done = term or trunc
        else:
            obs, rew, done, info = step_result
            term, trunc = done, False
        
        try:
            # Try getting ALE from unwrapped env (modern Gym)
            ram_state = unwrap(self.env).ale.getRAM()
        except:
             # Fallback if unwrapped logic fails or different structure
             raise RuntimeError("Could not access ALE RAM")

        for name, properties in MontezumaInfoWrapper.ram_map.items():
            info[name] = ram_state[properties['index']]
        pos = (info['x'], info['y'], info['room'])
        self.visited.add(pos)
        self.visited_rooms.add(info["room"])
        if done:
            info['mz_episode'] = dict(pos_count=len(self.visited),
                                      visited_rooms=copy(self.visited_rooms))
            self.visited.clear()
            self.visited_rooms.clear()
        
        if is_modern_gym:
            return obs, rew, term, trunc, info
        else:
            return obs, rew, done, info

    def reset(self, **kwargs: Any) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
        return self.env.reset(**kwargs)

class MarioXReward(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.current_level: List[int] = [0, 0]
        self.visited_levels: Set[Tuple[int, int]] = set()
        self.visited_levels.add(tuple(self.current_level)) # type: ignore
        self.current_max_x = 0.

    def reset(self, **kwargs: Any) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
        res = self.env.reset(**kwargs)
        if isinstance(res, tuple):
            ob = res[0]
        else:
            ob = res
        self.current_level = [0, 0]
        self.visited_levels = set()
        self.visited_levels.add(tuple(self.current_level)) # type: ignore
        self.current_max_x = 0.
        return res

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        step_result = self.env.step(action)
        if len(step_result) == 5:
            ob, reward, term, trunc, info = step_result
            done = term or trunc
        else:
            ob, reward, done, info = step_result
            term, trunc = done, False
            
        levellow = info.get("levelLo", 0)
        levelhigh = info.get("levelHi", 0)
        xscrollHi = info.get("xscrollHi", 0)
        xscrollLo = info.get("xscrollLo", 0)
        
        currentx = xscrollHi * 256 + xscrollLo
        new_level = [levellow, levelhigh]
        if new_level != self.current_level:
            self.current_level = new_level
            self.current_max_x = 0.
            reward = 0.
            self.visited_levels.add(tuple(self.current_level)) # type: ignore
        else:
            if currentx > self.current_max_x:
                delta = currentx - self.current_max_x
                self.current_max_x = currentx
                reward = delta
            else:
                reward = 0.
        if done:
            info["levels"] = copy(self.visited_levels)
            info["retro_episode"] = dict(levels=copy(self.visited_levels))
            
        if is_modern_gym:
            return ob, reward, term, trunc, info
        else:
            return ob, reward, done, info

class LimitedDiscreteActions(gym.ActionWrapper):
    KNOWN_BUTTONS = {"A", "B"}
    KNOWN_SHOULDERS = {"L", "R"}

    def __init__(self, env: gym.Env, all_buttons: List[str], whitelist: Set[str] = KNOWN_BUTTONS | KNOWN_SHOULDERS):
        super().__init__(env)
        self._num_buttons = len(all_buttons)
        button_keys = {i for i in range(len(all_buttons)) if all_buttons[i] in whitelist & self.KNOWN_BUTTONS}
        buttons = [(), *zip(button_keys), *itertools.combinations(button_keys, 2)]
        shoulder_keys = {i for i in range(len(all_buttons)) if all_buttons[i] in whitelist & self.KNOWN_SHOULDERS}
        shoulders = [(), *zip(shoulder_keys), *itertools.permutations(shoulder_keys, 2)]
        arrows = [(), (4,), (5,), (6,), (7,)]  # (), up, down, left, right
        acts = []
        acts += arrows
        acts += buttons[1:]
        acts += [a + b for a in arrows[-2:] for b in buttons[1:]]
        self._actions = acts
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a: int) -> np.ndarray:
        mask = np.zeros(self._num_buttons)
        for i in self._actions[a]:
            mask[i] = 1
        return mask

class FrameSkip(gym.Wrapper):
    def __init__(self, env: gym.Env, n: int):
        super().__init__(env)
        self.n = n

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        terminated = False
        truncated = False
        totrew = 0.0
        info: Dict[str, Any] = {}
        
        for _ in range(self.n):
            step_result = self.env.step(action)
            if len(step_result) == 5:
                ob, rew, term, trunc, info_step = step_result
                done = term or trunc
            else:
                ob, rew, done, info_step = step_result
                term, trunc = done, False
            
            info.update(info_step)
            totrew += rew
            if done:
                terminated = term
                truncated = trunc
                break
                
        if is_modern_gym:
            return ob, totrew, terminated, truncated, info
        else:
            return ob, totrew, terminated or truncated, info

class OneChannel(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, crop: bool = True):
        self.crop = crop
        super(OneChannel, self).__init__(env)
        # Assuming input is uint8
        # shape is (84, 84, 1) usually based on usage
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs[:, :, 2:3]

class RetroALEActions(gym.ActionWrapper):
    def __init__(self, env: gym.Env, all_buttons: List[str], n_players: int = 1):
        super().__init__(env)
        self.n_players = n_players
        self._num_buttons = len(all_buttons)
        bs = [-1, 0, 4, 5, 6, 7]
        actions: List[List[int]] = []

        def update_actions(old_actions: List[List[int]], offset: int = 0) -> List[List[int]]:
            actions = []
            for b in old_actions:
                for button in bs:
                    action = []
                    action.extend(b)
                    if button != -1:
                        action.append(button + offset)
                    actions.append(action)
            return actions

        current_actions: List[List[int]] = [[]]
        for i in range(self.n_players):
            current_actions = update_actions(current_actions, i * self._num_buttons)
        self._actions = current_actions
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a: int) -> np.ndarray:
        mask = np.zeros(self._num_buttons * self.n_players)
        for i in self._actions[a]:
            mask[i] = 1
        return mask

class NoReward(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        step_result = self.env.step(action)
        if len(step_result) == 5:
            ob, rew, term, trunc, info = step_result
            return ob, 0.0, term, trunc, info
        else:
            ob, rew, done, info = step_result
            return ob, 0.0, done, info

def make_mario_env(crop: bool = True, frame_stack: bool = True, clip_rewards: bool = False) -> gym.Env:
    assert clip_rewards is False
    import gym
    import retro
    # Modern FrameStack is in gym.wrappers or gym.wrappers.frame_stack
    # We use gym.wrappers.FrameStack which is standard.
    # Note: baselines.common.atari_wrappers.FrameStack is slightly different (lazy frames), 
    # but here we likely want standard behavior or to check if we need lazy frames.
    # The original code imported from baselines. 
    # To be exact, we should probably stick to standard gym FrameStack 
    # OR implement the LazyFrames one if memory is an issue. 
    # Given we are single-gpu, standard Gym FrameStack is likely fine.
    from gym.wrappers import FrameStack

    # gym.undo_logger_setup()
    env = retro.make('SuperMarioBros-Nes', 'Level1-1')
    buttons = env.BUTTONS
    env = MarioXReward(env)
    env = FrameSkip(env, 4)
    env = ProcessFrame84(env, crop=crop)
    if frame_stack:
        env = FrameStack(env, 4)
    env = LimitedDiscreteActions(env, buttons)
    return env

def make_multi_pong(frame_stack: bool = True) -> gym.Env:
    import gym
    import retro
    from gym.wrappers import FrameStack
    # gym.undo_logger_setup()
    game_env = env = retro.make('Pong-Atari2600', players=2)
    env = RetroALEActions(env, game_env.BUTTONS, n_players=2)
    env = NoReward(env)
    env = FrameSkip(env, 4)
    env = ProcessFrame84(env, crop=False)
    if frame_stack:
        env = FrameStack(env, 4)
    return env

def make_robo_pong(frame_stack: bool = True) -> gym.Env:
    from gym.wrappers import FrameStack
    import roboenvs as robo

    env = robo.make_robopong()
    env = robo.DiscretizeActionWrapper(env, 2)
    env = robo.MultiDiscreteToUsual(env)
    env = OneChannel(env)
    if frame_stack:
        env = FrameStack(env, 4)

    env = AddRandomStateToInfo(env)
    return env

def make_robo_hockey(frame_stack: bool = True) -> gym.Env:
    from gym.wrappers import FrameStack
    import roboenvs as robo

    env = robo.make_robohockey()
    env = robo.DiscretizeActionWrapper(env, 2)
    env = robo.MultiDiscreteToUsual(env)
    env = OneChannel(env)
    if frame_stack:
        env = FrameStack(env, 4)
    env = AddRandomStateToInfo(env)
    return env

class VecEnvAdapter:
    """
    Adapts a gym.vector.VectorEnv (which returns 5 values in step)
    to the interface expected by rollouts.py (obs, rew, done, info).
    """
    def __init__(self, vector_env: Any):
        self.vector_env = vector_env
        self.num_envs = vector_env.num_envs
        self.observation_space = vector_env.observation_space
        self.action_space = vector_env.action_space

    def reset(self) -> Any:
        obs, _ = self.vector_env.reset()
        return obs

    def step_async(self, actions: Any) -> None:
        self.vector_env.step_async(actions)

    def step_wait(self) -> Tuple[Any, Any, Any, Any]:
        result = self.vector_env.step_wait()
        # Modern gym vector envs return: obs, rews, terms, truncs, infos
        # We squash terms/truncs to done for compatibility
        if len(result) == 5:
            obs, rews, terms, truncs, infos = result
            dones = terms | truncs
            return obs, rews, dones, infos
        return result
    
    def close(self) -> None:
        self.vector_env.close()