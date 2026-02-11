import pickle
from pathlib import Path
from collections import deque
from typing import Any, NamedTuple, Union, Dict, Sequence

import gymnasium as gym
import numpy as np
import cv2
import torch
from dm_env import StepType, specs

import agxcave
from agxcave.agxenvs.utils.parse_cfg import parse_env_cfg

from agx_src.demo_reader import resize_chw_image, standardize_to_agxenv


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class TimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    rgb_obs: Any
    low_dim_obs: Any
    demo: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    rgb_obs: Any
    low_dim_obs: Any
    action: Any
    demo: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ExtendedTimeStepWrapper:
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            rgb_obs=time_step.rgb_obs,
            low_dim_obs=time_step.low_dim_obs,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward,
            discount=time_step.discount,
            demo=time_step.demo,
        )

    def low_dim_observation_spec(self):
        return self._env.low_dim_observation_spec()

    def rgb_observation_spec(self):
        return self._env.rgb_observation_spec()

    def low_dim_raw_observation_spec(self):
        return self._env.low_dim_raw_observation_spec()

    def rgb_raw_observation_spec(self):
        return self._env.rgb_raw_observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def render(self):
        return self._env.render()

    def __getattr__(self, name):
        return getattr(self._env, name)


class AGXEnv:
    def __init__(
        self,
        task_name: str,
        episode_length: int,
        frame_stack: int,
        dataset_root: str,
        camera_shape: tuple[int, int] = (84, 84),
        camera_keys: Sequence[str] = ("rgb", "depth"),
        stone_height_threshold: float = 1.5,
        headless: bool = True,
        reward_type: str = "sparse",
        state_based_only: bool = False,
    ):
        self._task_name = task_name
        self._episode_length = int(episode_length)
        self._frame_stack = int(frame_stack)
        self._dataset_root = str(dataset_root)
        self._camera_shape = (int(camera_shape[0]), int(camera_shape[1]))
        self._camera_keys = tuple(camera_keys)
        self._stone_height_threshold = float(stone_height_threshold)
        self._frames = {"rgb": deque([], maxlen=self._frame_stack)}
        self._reward_type = reward_type
        self._state_based_only = state_based_only
        self._last_termination_info = {}


        cfg = parse_env_cfg(
            task_name,
            device="cuda",
            headless=headless,
            render_mode=None,
        )

        self._env = gym.make(task_name, cfg=cfg, agx_args=[])

        self.action_space = self._env.action_space
        self._low_dim_raw_dim = 10 # 3 + 3 + 3 + 1

        self.low_dim_raw_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._low_dim_raw_dim,), dtype=np.float32
        )
        self.low_dim_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._low_dim_raw_dim * self._frame_stack,),
            dtype=np.float32,
        )

        self.rgb_raw_observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(1, 3, *self._camera_shape),   # raw = 1 view, 3 channels
            dtype=np.uint8,
        )
        self.rgb_observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(1, 3 * self._frame_stack, *self._camera_shape),  # stacked in channels
            dtype=np.uint8,
        )

        self._low_dim_obses = deque([], maxlen=self._frame_stack)
        self._frames = {k: deque([], maxlen=self._frame_stack) for k in self._camera_keys}
        self._step_counter = 0
        self._last_render = None

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    # replay buffer
    def low_dim_observation_spec(self):
        return specs.Array(self.low_dim_observation_space.shape, np.float32, "low_dim_obs")

    def low_dim_raw_observation_spec(self):
        return specs.Array(self.low_dim_raw_observation_space.shape, np.float32, "low_dim_obs")

    def rgb_observation_spec(self):
        return specs.Array(self.rgb_observation_space.shape, np.uint8, "rgb_obs")

    def rgb_raw_observation_spec(self):
        return specs.Array(self.rgb_raw_observation_space.shape, np.uint8, "rgb_obs")

    def action_spec(self):
        return specs.Array(self.action_space.shape, np.float32, "action")

    def render(self):
        if self._state_based_only:
            return None
        if self._last_render is None:
            return np.zeros((*self._camera_shape, 3), dtype=np.uint8)
        return self._last_render

    def _build_low_dim_raw(self, obs_dict):
        state = _to_numpy(obs_dict["policy"]).reshape(-1).astype(np.float32)[:3] # only relevant ones
        bucket_pos = _to_numpy(obs_dict["bucket"]).reshape(-1).astype(np.float32)
        cabin_pos = _to_numpy(obs_dict["cabin_position"]).reshape(-1).astype(np.float32)
        cabin_pitch = _to_numpy(obs_dict["cabin_pitch"]).reshape(-1).astype(np.float32)

        low = np.concatenate([state, bucket_pos, cabin_pos, cabin_pitch], axis=0)
        assert low.shape[0] == self._low_dim_raw_dim, low.shape
        return low


    def _build_rgb_raw(self, obs_dict):
        # support resized images
        img = _to_numpy(obs_dict["camera"]["rgb"])
        if img.shape[-2:] != self._camera_shape:
            img = img.astype(np.uint8, copy=False)
            img = resize_chw_image(img, self._camera_shape)
        self._last_render = img
        return img[None, ...] # (1,3,H,W)

    def _extract_obs(self, obs_dict):
        low_raw = self._build_low_dim_raw(obs_dict)

        if len(self._low_dim_obses) == 0:
            for _ in range(self._frame_stack):
                self._low_dim_obses.append(low_raw)
        else:
            self._low_dim_obses.append(low_raw)
        
        low_stacked = np.concatenate(list(self._low_dim_obses), axis=0)  # (D*frame_stack,)

        if self._state_based_only:
            rgb_stacked = np.zeros(self.rgb_observation_space.shape, dtype=np.uint8)
            return {"low_dim_obs": low_stacked, "rgb_obs": rgb_stacked}

        rgb_raw = self._build_rgb_raw(obs_dict)
        # keep a single deque of rgb frames (each is (3,H,W))
        if len(self._frames["rgb"]) == 0:
            for _ in range(self._frame_stack):
                self._frames["rgb"].append(rgb_raw[0])
        else:
            self._frames["rgb"].append(rgb_raw[0])

        

        # channel-concat: list of (3,H,W) -> (3*frame_stack,H,W) then add view dim -> (1, 3*frame_stack, H, W)
        rgb_stacked_chw = np.concatenate(list(self._frames["rgb"]), axis=0)
        rgb_stacked = rgb_stacked_chw[None, ...]

        return {"low_dim_obs": low_stacked, "rgb_obs": rgb_stacked}
    
    def _compute_reward(self, obs_dict):
        if self._reward_type == "sparse":
            if _to_numpy(obs_dict["stone"]).reshape(-1)[2] >= self._stone_height_threshold:
                return 1
            else:
                return 0
        else:
            # TODO
            z = _to_numpy(obs_dict["stone"]).reshape(-1)[2]

            # distance to target height
            dist = z - self._stone_height_threshold
            reward = -abs(dist)

            # If proper height reached
            if z >= 1.5:
                reward += 10

        return reward
        
    def reset(self, **kwargs):
        self._low_dim_obses.clear()
        for q in self._frames.values():
            q.clear()

        obs, info = self._env.reset(**kwargs)
        obs = self._extract_obs(obs)
        self._step_counter = 0
        self._last_termination_info = {}

        return TimeStep(
            rgb_obs=obs["rgb_obs"],
            low_dim_obs=obs["low_dim_obs"],
            step_type=StepType.FIRST,
            reward=np.array([0.0], dtype=np.float32),
            discount=np.array([1.0], dtype=np.float32),
            demo=np.array([0.0], dtype=np.float32),
        )

    def step(self, action: np.ndarray):
        obs, _, terminated, truncated, info = self._env.step(action)
        obs_dict = obs
        obs = self._extract_obs(obs_dict)
        self._step_counter += 1

        if self._step_counter >= self._episode_length:
            truncated = True

        done = bool(terminated or truncated)
        step_type = StepType.LAST if done else StepType.MID

        # sparse reward
        reward = self._compute_reward(obs_dict)

        # verify if we actually get terminted correctly from agx
        # if self._reward_type == "sparse" and reward == 0:
        #     terminated = True

        # discount=0 only on success
        # potential fix to the issue where agent learns to trigger unsafe terminations as reward hack
        extras = info.get("extras", {})
        self._last_termination_info = {
            k.replace("Episode_Termination/", ""): int(v)
            for k, v in extras.items()
            if k.startswith("Episode_Termination/")
        }
        success_flag = bool(extras.get("Episode_Termination/stone_height_termination", 0))
        if terminated and success_flag:
            discount = 0.0
        else:
            discount = 1.0

        return TimeStep(
            rgb_obs=obs["rgb_obs"],
            low_dim_obs=obs["low_dim_obs"],
            step_type=step_type,
            reward=np.array([reward], dtype=np.float32),
            discount=np.array([discount], dtype=np.float32),
            demo=np.array([0.0], dtype=np.float32),
        )

    def get_demos(self, num_demos: int):
        demo_dir = Path(self._dataset_root)
        pkls = sorted(demo_dir.glob("*.pkl"))
        demos = []
        for i, p in enumerate(pkls[:num_demos]):
            with p.open("rb") as f:
                traj = pickle.load(f)
            traj = [standardize_to_agxenv(s) for s in traj]
            demo = self.convert_demo_to_timesteps(traj)
            yield demo
        #     demos.append(demo)
        # return demos

    def convert_demo_to_timesteps(self, traj: list[dict]):
        timesteps = []
        self._low_dim_obses.clear()
        for q in self._frames.values():
            q.clear()
        T = len(traj)
        if T < 2:
            return None

        for i in range(T):
            obs_dict = traj[i]
            obs = self._extract_obs(obs_dict)

            if i == 0:
                action = np.zeros(self.action_space.shape, dtype=np.float32)
                step_type = StepType.FIRST
                reward = 0.0
                discount = 1.0
            else:
                action = _to_numpy(traj[i - 1]["action"]).reshape(-1).astype(np.float32)
                if i == T - 1:
                    step_type = StepType.LAST
                    reward = self._compute_reward(obs_dict)
                    if self._reward_type != "sparse":
                        if _to_numpy(obs_dict["stone"]).reshape(-1)[2] >= self._stone_height_threshold:
                            reward = 200.0
                    discount = 0.0
                else:
                    step_type = StepType.MID
                    reward = self._compute_reward(obs_dict)
                    if self._reward_type == "sparse":
                        discount = 1.0
                    else:
                        discount = 0.0 if reward > 0 else 1.0

            timesteps.append(
                ExtendedTimeStep(
                    rgb_obs=obs["rgb_obs"],
                    low_dim_obs=obs["low_dim_obs"],
                    step_type=step_type,
                    action=action,
                    reward=np.array([reward], dtype=np.float32),
                    discount=np.array([discount], dtype=np.float32),
                    demo=np.array([1.0], dtype=np.float32),
                )
            )
        return timesteps


def make(
    task_name,
    episode_length,
    frame_stack,
    dataset_root,
    camera_shape,
    stone_height_threshold,
    camera_keys=("rgb", "depth"),
    reward_type="sparse",
    state_based_only=False,
):
    env = AGXEnv(
        task_name=task_name,
        episode_length=episode_length,
        frame_stack=frame_stack,
        dataset_root=dataset_root,
        camera_shape=tuple(camera_shape),
        camera_keys=camera_keys,
        stone_height_threshold=stone_height_threshold,
        headless=True,
        reward_type=reward_type,
        state_based_only=state_based_only
    )
    return ExtendedTimeStepWrapper(env)

