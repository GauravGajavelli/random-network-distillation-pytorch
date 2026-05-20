"""Environment wrappers and subprocess workers.

Migrated to the gymnasium API. The Pipe-based subprocess protocol sends a
single 6-tuple per step:
    [history, reward, force_done, real_done, log_reward, extras]
where ``extras`` is a dict that may contain keys like ``inventory_vec``,
``died``, ``reached_goal``, ``visited_rooms``. Consumers ignore unfamiliar
keys. Backwards compatibility note: the existing 5-tuple receive code in
``train.py`` has been updated to match.
"""
import os
from abc import abstractmethod
from collections import deque
from copy import copy

import cv2
import numpy as np
from PIL import Image

import gymnasium as gym

from torch.multiprocessing import Process

from config import default_config


train_method = default_config['TrainMethod']
max_step_per_episode = int(default_config['MaxStepPerEpisode'])


class Environment(Process):
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def pre_proc(self, x):
        pass

    @abstractmethod
    def get_init_state(self, x):
        pass


def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    else:
        return env


# ===================================================================
# Atari-side wrappers (gymnasium API)
# ===================================================================


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, is_render, skip=4):
        super().__init__(env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        self.is_render = is_render

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        obs = None
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if self.is_render:
                self.env.render()
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env, room_address):
        super().__init__(env)
        self.room_address = room_address
        self.visited_rooms = set()

    def get_current_room(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())
        info.setdefault('episode', {}).update(visited_rooms=copy(self.visited_rooms))
        if terminated or truncated:
            self.visited_rooms.clear()
        return obs, rew, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class AtariEnvironment(Environment):
    def __init__(self, env_id, is_render, env_idx, child_conn,
                 history_size=4, h=84, w=84, life_done=True,
                 sticky_action=True, p=0.25, **kwargs):
        super().__init__()
        self.daemon = True
        try:
            import ale_py
            gym.register_envs(ale_py)
        except ImportError:
            pass
        self.env = MaxAndSkipEnv(gym.make(env_id), is_render)
        if 'Montezuma' in env_id:
            self.env = MontezumaInfoWrapper(self.env, room_address=3)
        self.env_id = env_id
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.sticky_action = sticky_action
        self.last_action = 0
        self.p = p

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.h = h
        self.w = w
        self.reset()

    def run(self):
        super().run()
        while True:
            action = self.child_conn.recv()
            if 'Breakout' in self.env_id:
                action += 1
            if self.sticky_action and np.random.rand() <= self.p:
                action = self.last_action
            self.last_action = action

            s, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)
            if max_step_per_episode < self.steps:
                done = True
            log_reward = float(reward)
            force_done = done

            self.history[:3, :, :] = self.history[1:, :, :]
            self.history[3, :, :] = self.pre_proc(s)
            self.rall += reward
            self.steps += 1

            extras = {'visited_rooms': info.get('episode', {}).get('visited_rooms', set())}

            if done:
                self.recent_rlist.append(self.rall)
                print(f"[Episode {self.episode}({self.env_idx})] "
                      f"Step: {self.steps}  Reward: {self.rall}  "
                      f"Recent Reward: {np.mean(self.recent_rlist):.2f}  "
                      f"Visited Room: [{extras['visited_rooms']}]")
                self.history = self.reset()

            self.child_conn.send(
                [self.history[:, :, :], reward, force_done, done, log_reward, extras])

    def reset(self):
        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.rall = 0
        s, _info = self.env.reset()
        self.get_init_state(self.pre_proc(s))
        return self.history[:, :, :]

    def pre_proc(self, X):
        X = np.array(Image.fromarray(X).convert('L')).astype('float32')
        return cv2.resize(X, (self.h, self.w))

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)


# ===================================================================
# MiniGrid wrappers and environment process
# ===================================================================

# Object/color codes match minigrid.core.constants (re-declared here to avoid
# importing minigrid at module load — the import happens inside subprocesses).
OBJECT_TO_IDX = {
    'unseen': 0, 'empty': 1, 'wall': 2, 'floor': 3, 'door': 4,
    'key': 5, 'ball': 6, 'box': 7, 'goal': 8, 'lava': 9, 'agent': 10,
}
COLOR_TO_IDX = {
    'red': 0, 'green': 1, 'blue': 2, 'purple': 3, 'yellow': 4, 'grey': 5,
}
INVENTORY_DIM = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX)


def encode_carrying(carrying):
    """Encode the agent's carried object as a fixed-length one-hot vector."""
    vec = np.zeros(INVENTORY_DIM, dtype=np.float32)
    if carrying is None:
        return vec
    obj_type = getattr(carrying, 'type', None)
    obj_color = getattr(carrying, 'color', None)
    if obj_type in OBJECT_TO_IDX:
        vec[OBJECT_TO_IDX[obj_type]] = 1.0
    if obj_color in COLOR_TO_IDX:
        vec[len(OBJECT_TO_IDX) + COLOR_TO_IDX[obj_color]] = 1.0
    return vec


class NoisyTVWrapper(gym.Wrapper):
    """Moving variable-size random pixel patch overlaid on the RGB observation.

    Apply *after* a wrapper that produces an HxWxC uint8 image (e.g.
    minigrid's RGBImgPartialObsWrapper + ImgObsWrapper).
    """

    def __init__(self, env, tv_on=True, tv_max_size=9, p_move=1.0, seed=0):
        super().__init__(env)
        self.tv_on = tv_on
        self.tv_max_size = tv_max_size
        self.p_move = p_move
        self._rng = np.random.RandomState(seed)
        self._cur_x = 0
        self._cur_y = 0
        self._cur_s = 3

    def _inject(self, obs):
        if not self.tv_on:
            return obs
        if self._rng.rand() <= self.p_move:
            sizes = [s for s in (3, 5, 7, 9) if s <= self.tv_max_size]
            self._cur_s = int(self._rng.choice(sizes))
            self._cur_x = int(self._rng.randint(0, max(1, obs.shape[1] - self._cur_s)))
            self._cur_y = int(self._rng.randint(0, max(1, obs.shape[0] - self._cur_s)))
        s = self._cur_s
        patch = self._rng.randint(0, 256, (s, s, obs.shape[2])).astype(obs.dtype)
        obs = obs.copy()
        obs[self._cur_y:self._cur_y + s, self._cur_x:self._cur_x + s] = patch
        return obs

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        return self._inject(obs), r, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._inject(obs), info


class MiniGridEnvironment(Environment):
    """MiniGrid worker process.

    Sends [history, reward, force_done, real_done, log_reward, extras] per step.
    ``extras`` carries: inventory_vec, died, reached_goal.

    LavaCrossing semantics: stepping on lava terminates the episode with
    reward 0 (no positive reward). We classify any termination without a
    positive reward as a death for logging purposes.
    """

    def __init__(self, env_id, is_render, env_idx, child_conn,
                 history_size=4, h=84, w=84, life_done=False,
                 sticky_action=False, p=0.0,
                 tv_on=False, tv_max_size=9, p_move=1.0,
                 use_inventory=False, tile_size=8, **kwargs):
        super().__init__()
        self.daemon = True

        import minigrid  # noqa: F401  (registers envs)
        from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

        base = gym.make(env_id)
        base = RGBImgPartialObsWrapper(base, tile_size=tile_size)
        base = ImgObsWrapper(base)
        if tv_on:
            base = NoisyTVWrapper(base, tv_on=True, tv_max_size=tv_max_size,
                                  p_move=p_move, seed=env_idx)
        self.env = base
        self.env_id = env_id
        self.env_idx = env_idx
        self.is_render = is_render
        self.use_inventory = use_inventory

        self.steps = 0
        self.episode = 0
        self.rall = 0.0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.last_action = 0
        self.sticky_action = sticky_action
        self.p = p

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w], dtype=np.float32)
        self.h = h
        self.w = w
        self.inventory_vec = np.zeros(INVENTORY_DIM, dtype=np.float32)
        self.died = False
        self.reached_goal = False
        self.reset()

    def run(self):
        super().run()
        while True:
            action = self.child_conn.recv()
            if self.sticky_action and np.random.rand() <= self.p:
                action = self.last_action
            self.last_action = action

            s, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)
            if max_step_per_episode < self.steps:
                done = True
            log_reward = float(reward)
            force_done = done

            # Episode-end classification for logging
            self.reached_goal = bool(terminated and reward > 0)
            self.died = bool(terminated and reward <= 0)

            self.history[:3, :, :] = self.history[1:, :, :]
            self.history[3, :, :] = self.pre_proc(s)

            if self.use_inventory:
                self.inventory_vec = encode_carrying(self.env.unwrapped.carrying)

            self.rall += reward
            self.steps += 1

            extras = {
                'inventory_vec': self.inventory_vec.copy(),
                'died': self.died if done else False,
                'reached_goal': self.reached_goal if done else False,
            }

            if done:
                self.recent_rlist.append(self.rall)
                print(f"[Episode {self.episode}({self.env_idx})] "
                      f"Step: {self.steps}  Reward: {self.rall:.3f}  "
                      f"Recent: {np.mean(self.recent_rlist):.3f}  "
                      f"died={self.died} goal={self.reached_goal}")
                self.history = self.reset()
                # clear after sending: the post-reset state is what we send next loop
                self.died = False
                self.reached_goal = False

            self.child_conn.send(
                [self.history[:, :, :], reward, force_done, done, log_reward, extras])

    def reset(self):
        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.rall = 0.0
        self.died = False
        self.reached_goal = False
        s, _info = self.env.reset()
        if self.use_inventory:
            self.inventory_vec = encode_carrying(self.env.unwrapped.carrying)
        else:
            self.inventory_vec = np.zeros(INVENTORY_DIM, dtype=np.float32)
        self.get_init_state(self.pre_proc(s))
        return self.history[:, :, :]

    def pre_proc(self, X):
        if X.ndim == 3:
            x = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
        else:
            x = X
        x = cv2.resize(x, (self.h, self.w))
        return x.astype('float32')

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)


# ===================================================================
# Mario stub — gym-super-mario-bros has not migrated to gymnasium cleanly.
# Kept as NotImplementedError so the existing config keys don't import-fail.
# ===================================================================


class MarioEnvironment(Environment):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "MarioEnvironment is not supported under the gymnasium migration. "
            "Use EnvType=atari or EnvType=minigrid.")

    def run(self):
        pass

    def reset(self):
        pass

    def pre_proc(self, x):
        pass

    def get_init_state(self, x):
        pass
