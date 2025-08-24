"""Environment preprocessing wrappers (refactored from utils/preprocess_env.py).

Only the actively used wrappers were retained to reduce maintenance burden:
 - RepeatAction
 - RescaleObservation
 - ReshapeObservation (channel-first transpose variant)
 - FilterFromDict
 - ColorTransformObservation

If you need any of the previously commented legacy wrappers, retrieve them from git history.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, WrapperObsType
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box


class ColorTransformObservation(ObservationWrapper):
    """Apply a simple color tint to RGB observations.

    Allowed colors: standard (no-op), red, green, blue.
    """

    def __init__(self, env, color: str = "green"):
        super().__init__(env)
        assert color in ["standard", "green", "red", "blue"], (
            "Color must be 'standard', 'green', 'red', or 'blue'."
        )
        self.color = color
        assert isinstance(self.observation_space, Box), "Observation space must be Box"
        assert len(self.observation_space.shape) == 3, "Expect (H,W,C) observation"
        assert self.observation_space.shape[2] == 3, "Expect 3 channel RGB"

    def observation(self, obs):  # type: ignore[override]
        if self.color == "standard":
            return obs
        tint = np.zeros_like(obs, dtype=np.float32)
        if self.color == "red":
            tint[..., 0] = 1.0
        elif self.color == "green":
            tint[..., 1] = 1.0
        elif self.color == "blue":
            tint[..., 2] = 1.0
        blend = 0.2
        obs_tinted = (1 - blend) * obs + blend * tint * 255
        return np.clip(obs_tinted, 0, 255).astype(np.uint8)


class TransformObservation(
    gym.ObservationWrapper[WrapperObsType, ActType, ObsType], gym.utils.RecordConstructorArgs
):
    """Apply a function to observations (thin re-implementation)."""

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        func: Callable[[ObsType], Any],
        observation_space: Optional[gym.Space[WrapperObsType]] = None,
    ):
        gym.utils.RecordConstructorArgs.__init__(self, func=func, observation_space=observation_space)
        gym.ObservationWrapper.__init__(self, env)
        if observation_space is not None:
            self.observation_space = observation_space
        self.func = func

    def observation(self, observation: ObsType) -> Any:  # type: ignore[override]
        return self.func(observation)


class ReshapeObservation(TransformObservation):
    """Transpose HWC -> CHW (expects 3D image, user supplies target shape)."""

    def __init__(self, env: gym.Env[ObsType, ActType], shape: Union[int, Tuple[int, ...]]):
        assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(shape, tuple)
        assert np.prod(env.observation_space.shape) == np.prod(shape)
        new_obs_space = spaces.Box(
            low=np.reshape(np.ravel(env.observation_space.low), shape),
            high=np.reshape(np.ravel(env.observation_space.high), shape),
            shape=shape,
            dtype=env.observation_space.dtype,
        )
        super().__init__(env, func=lambda obs: np.transpose(obs, (2, 0, 1)), observation_space=new_obs_space)


class RescaleObservation(TransformObservation):
    """Rescale uint8 [0,255] to [0,1]."""

    def __init__(self, env: gym.Env[ObsType, ActType], rescale_value: float = 255.0):
        assert isinstance(env.observation_space, spaces.Box)
        new_obs_space = spaces.Box(
            low=env.observation_space.low / rescale_value,
            high=env.observation_space.high / rescale_value,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )
        super().__init__(env, func=lambda obs: obs / rescale_value, observation_space=new_obs_space)


class FilterFromDict(gym.ObservationWrapper):
    """Return only the 'image' key from a dict observation space (minigrid style)."""

    def __init__(self, env=None, key: str | None = None):
        super().__init__(env)
        if key is None:
            key = "image"
        self.key = key
        self.observation_space = env.observation_space.spaces[self.key]

    def observation(self, observation):  # type: ignore[override]
        return observation[self.key]


class RepeatAction(gym.Wrapper):
    """Repeat the same action 'repeat' times accumulating reward (modern Gymnasium API)."""

    def __init__(self, env=None, repeat: int = 4, clip_rewards: bool = False, fire_first: bool = False):
        super().__init__(env)
        self.repeat = repeat
        self.clip_rewards = clip_rewards
        self.fire_first = fire_first

    def step(self, action):  # type: ignore[override]
        total_reward = 0.0
        terminated = truncated = False
        last_obs = None
        info = {}
        for _ in range(self.repeat):
            obs, reward, term, trunc, info = self.env.step(action)
            last_obs = obs
            if self.clip_rewards:
                reward = np.sign(reward)
            total_reward += reward
            terminated |= term
            truncated |= trunc
            if term or trunc:
                break
        assert last_obs is not None
        return last_obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        if self.fire_first:
            if "FIRE" in self.env.unwrapped.get_action_meanings():
                fire_idx = self.env.unwrapped.get_action_meanings().index("FIRE")
                obs, _, _, _, _ = self.env.step(fire_idx)
        return obs, info
