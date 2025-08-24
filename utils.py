import math
import os
import random
from collections import deque
import os
import sys
import numpy as np

# Ensure a headless MuJoCo GL backend when DISPLAY is not available
if 'MUJOCO_GL' not in os.environ:
    if sys.platform.startswith('linux') and not os.getenv('DISPLAY'):
        os.environ['MUJOCO_GL'] = 'egl'
# import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from datetime import datetime
import subprocess
import json
import glob
import gymnasium as gym


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # gain = nn.init.calculate_gain('relu')
        # nn.init.orthogonal_(m.weight.data, gain)
        # if hasattr(m.bias, 'data'):
        #     m.bias.data.fill_(0.0)
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def flatten_batch_channels(obs):
    """Flatten a 4D tensor/array (B, C, H, W) into (B*C, H, W).

    Returns
    -------
    flat : same type as input (torch.Tensor or np.ndarray)
        The reshaped view/copy with combined batch and channel dimension.
    original_shape : tuple
        The original (B, C, H, W) shape for later restoration.

    Notes
    -----
    * For torch.Tensor the returned tensor is a view when possible (no data copy).
    * Use `unflatten_batch_channels` with the saved original_shape to revert.
    """
    if isinstance(obs, torch.Tensor):
        assert obs.dim() == 4, f"Expected 4D tensor (B,C,H,W), got {obs.shape}"
        B, C, H, W = obs.shape
        return obs.view(B * C, H, W), (B, C, H, W)
    elif isinstance(obs, np.ndarray):
        assert obs.ndim == 4, f"Expected 4D array (B,C,H,W), got {obs.shape}"
        B, C, H, W = obs.shape
        return obs.reshape(B * C, H, W), (B, C, H, W)
    else:
        raise TypeError("obs must be torch.Tensor or np.ndarray")


def unflatten_batch_channels(flat_obs, original_shape):
    """Inverse of `flatten_batch_channels`.

    Parameters
    ----------
    flat_obs : torch.Tensor | np.ndarray
        Tensor/array with shape (B*C, H, W).
    original_shape : tuple
        The original (B, C, H, W) shape tuple returned by `flatten_batch_channels`.

    Returns
    -------
    obs : same type as input
        Reshaped back to (B, C, H, W).
    """
    if not isinstance(original_shape, (tuple, list)) or len(original_shape) != 4:
        raise ValueError("original_shape must be a 4-tuple (B,C,H,W)")
    B, C, H, W = original_shape
    if isinstance(flat_obs, torch.Tensor):
        assert flat_obs.dim() == 3, f"Expected 3D tensor (B*C,H,W), got {flat_obs.shape}"
        assert flat_obs.shape[0] == B * C and flat_obs.shape[1] == H and flat_obs.shape[2] == W
        return flat_obs.view(B, C, H, W)
    elif isinstance(flat_obs, np.ndarray):
        assert flat_obs.ndim == 3, f"Expected 3D array (B*C,H,W), got {flat_obs.shape}"
        assert flat_obs.shape[0] == B * C and flat_obs.shape[1] == H and flat_obs.shape[2] == W
        return flat_obs.reshape(B, C, H, W)
    else:
        raise TypeError("flat_obs must be torch.Tensor or np.ndarray")


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype)
        # Safely capture max episode steps if available (avoid direct private attr access in Gymnasium)
        self._max_episode_steps = getattr(env, '_max_episode_steps', None)
        if self._max_episode_steps is None and getattr(env, 'spec', None) is not None:
            self._max_episode_steps = env.spec.max_episode_steps
        
    def reset(self, colour=None):
        obs = self.env.reset(colour=colour)
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)

    def custom_reset(self, frames):
        for _ in range(self._k):
            self._frames.append(frames[0])
        return self._get_obs()


class GymV21StepAPIWrapper(gym.Wrapper):
    """Wrap a Gymnasium env returning (terminated, truncated) into classic (done).

    step(action) -> obs, reward, done, info where done = terminated or truncated.
    reset(...) -> obs (drops info for backward compatibility with legacy code).
    """

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Retain termination semantics for downstream code that wants to distinguish time limits
        info = dict(info)  # shallow copy to avoid mutating upstream
        info.setdefault('terminated', terminated)
        info.setdefault('truncated', truncated)
        # Classic done signal (used by legacy replay buffers / logging)
        done = terminated or truncated
        return obs, reward, done, info

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        return obs



class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

def write_info(args, fp):
    try:
        data = {
            'timestamp': str(datetime.now()),
            'git': subprocess.check_output(["git", "describe", "--always"]).strip().decode(),
            'args': vars(args)
            }
    except:
        data = {
            'timestamp': str(datetime.now()),
            'args': vars(args)
        }
    with open(fp, 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))

def listdir(dir_path, filetype='jpg', sort=True):
    fpath = os.path.join(dir_path, f'*.{filetype}')
    fpaths = glob.glob(fpath, recursive=True)
    if sort:
        return sorted(fpaths)
    return



from env_utils.preprocess_env import (
    RepeatAction,
    RescaleObservation,
    ReshapeObservation,
    FilterFromDict,
    ColorTransformObservation
)

try:
    from stable_baselines3.common.atari_wrappers import (  # isort:skip
        ClipRewardEnv,
        EpisodicLifeEnv,
        FireResetEnv,
        MaxAndSkipEnv,
        NoopResetEnv,
    )
except ImportError:
    # Define lightweight placeholders that raise only if actually instantiated
    class _MissingWrapper:
        def __init__(self, *_, **__):
            raise ImportError(
                "stable-baselines3 not installed. Install with `pip install stable-baselines3` to use this wrapper."
            )

    ClipRewardEnv = EpisodicLifeEnv = FireResetEnv = MaxAndSkipEnv = NoopResetEnv = _MissingWrapper


class MergeStackChannelWrapper(gym.ObservationWrapper):
    """
    Converts observations shaped (stack, C, H, W) into (stack*C, H, W)
    so encoders expecting (channels, H, W) with stacked frames along channel dim work.
    If the observation is already 3D (C,H,W) it is passed through unchanged.
    """
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box), "Only Box spaces supported"
        old_space = env.observation_space
        if len(old_space.shape) == 4:
            s, c, h, w = old_space.shape
            new_shape = (s * c, h, w)
            low = old_space.low.reshape(new_shape)
            high = old_space.high.reshape(new_shape)
            self.observation_space = gym.spaces.Box(
                low=low,
                high=high,
                shape=new_shape,
                dtype=old_space.dtype
            )
        else:
            # pass-through (already (C,H,W))
            self.observation_space = old_space

    def observation(self, obs):
        if isinstance(obs, np.ndarray) and obs.ndim == 4:
            s, c, h, w = obs.shape
            return obs.reshape(s * c, h, w)
        # LazyFrames or other sequence types: convert to np.array first
        try:
            if hasattr(obs, 'shape') and len(obs.shape) == 4:
                s, c, h, w = obs.shape
                return np.asarray(obs).reshape(s * c, h, w)
        except Exception:
            pass
        return obs


# class ActionFloat32Wrapper(gym.ActionWrapper):
#     def action(self, action):
#         # Ensure torch tensors or lists become proper float32 numpy array
#         if isinstance(action, torch.Tensor):
#             action = action.detach().cpu().numpy()
#         action = np.asarray(action, dtype=np.float32)
#         # Flatten any extra leading dims (e.g. (1,3) -> (3,))
#         if action.ndim > 1:
#             action = action.reshape(-1)
#         return action

class ActionFloat32Wrapper(gym.ActionWrapper):
    def action(self, action):
        """
        Convert incoming action to a simple tuple of Python floats.
        Box2D (via SWIG) rejects numpy.float32 scalars for some setters (e.g. motorSpeed),
        raising: TypeError ... argument 2 of type 'float32'. Using native floats avoids this.
        """
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        # Normalize to 1D array first
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        # Return tuple of native Python floats
        return tuple(float(x) for x in action)
    
# def make_env_atari(
#     env,
#     seed=0,
#     rgb=True,
#     stack=0,
#     no_op=0,
#     action_repeat=0,
#     max_frames=False,
#     episodic_life=False,
#     clip_reward=False,
#     check_fire=True,
#     color_transform="standard",
#     filter_dict=None,
#     time_limit: int = 0,
#     idx=0,
#     capture_video=False,
#     run_name="",
# ):
#     def thunk(env=env):
#         # print('Observation space: ', env.observation_space, 'Action space: ', env.action_space)
#         # env = gym.make(env_id)
#         # env = CarRacing(continuous=False, background='red')
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         if capture_video:
#             if idx == 0:
#                 env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         if no_op > 0:
#             env = NoopResetEnv(env, noop_max=30)
#         if action_repeat > 0:
#             if max_frames:
#                 env = MaxAndSkipEnv(env, skip=action_repeat if action_repeat > 1 else 4)
#             else:
#                 env = RepeatAction(env, repeat=action_repeat)
#         if episodic_life:
#             env = EpisodicLifeEnv(env)
#         if check_fire and "FIRE" in env.unwrapped.get_action_meanings():
#             env = FireResetEnv(env)
#         if clip_reward:
#             env = ClipRewardEnv(env)
#         if filter_dict:
#             env = FilterFromDict(env, filter_dict)
#         env = gym.wrappers.ResizeObservation(env, (84, 84))
#         if color_transform != "standard":
#             env = ColorTransformObservation(env, color=color_transform)
#         env = RescaleObservation(env, rescale_value=255.0)
#         if rgb:
#             #     env = PreprocessFrameRGB((84, 84, 3), env)  #
#             # env = ReshapeObservation(env, (3, 96, 96)) # replace with env.observation_space.shape[1],
            
#             # env = ReshapeObservation(env, (shape[2], shape[0], shape[1]))
#             env = ReshapeObservation(env, (3, 84, 84))
#         if not rgb:
#             # env = gym.wrappers.ResizeObservation(env, (84, 84))
#             env = gym.wrappers.GrayScaleObservation(env)
#         # env = NormalizeFrames(env)
#         # env = gym.wrappers.GrayScaleObservation(env)
#         if stack > 1:
#             env = gym.wrappers.FrameStack(env, stack)  # (4, 3, 84, 84)
#         if time_limit > 0:
#             env = gym.wrappers.TimeLimit(env, max_episode_steps=time_limit)
#         # env.seed(seed)
#         env.action_space.seed(seed)
#         env.observation_space.seed(seed)
#         return env

#     return thunk

def make_env_atari_single(
    env,
    seed=0,
    rgb=True,
    stack=0,
    no_op=0,
    action_repeat=0,
    max_frames=False,
    episodic_life=False,
    clip_reward=False,
    check_fire=True,
    color_transform="standard",
    filter_dict=None,
    time_limit: int = 0,
    idx=0,
    capture_video=False,
    run_name="",
):
    # print('Observation space: ', env.observation_space, 'Action space: ', env.action_space)
    # env = gym.make(env_id)
    # env = CarRacing(continuous=False, background='red')
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if capture_video:
        if idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    if no_op > 0:
        env = NoopResetEnv(env, noop_max=30)
    if action_repeat > 0:
        if max_frames:
            env = MaxAndSkipEnv(env, skip=action_repeat if action_repeat > 1 else 4)
        else:
            env = RepeatAction(env, repeat=action_repeat)
    if episodic_life:
        env = EpisodicLifeEnv(env)
    if check_fire and "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    if clip_reward:
        env = ClipRewardEnv(env)
    if filter_dict:
        env = FilterFromDict(env, filter_dict)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    if color_transform != "standard":
        env = ColorTransformObservation(env, color=color_transform)
    # env = RescaleObservation(env, rescale_value=255.0)
    if rgb:
        #     env = PreprocessFrameRGB((84, 84, 3), env)  #
        # env = ReshapeObservation(env, (3, 96, 96)) # replace with env.observation_space.shape[1],
        
        # env = ReshapeObservation(env, (shape[2], shape[0], shape[1]))
        env = ReshapeObservation(env, (3, 84, 84))
    if not rgb:
        # env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
    # env = NormalizeFrames(env)
    # env = gym.wrappers.GrayScaleObservation(env)
    if stack > 1:
        env = gym.wrappers.FrameStack(env, stack)  # (4, 3, 84, 84)
        env = MergeStackChannelWrapper(env)  # (4*3, 84, 84)
    if time_limit > 0:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=time_limit)
    env = ActionFloat32Wrapper(env)
    env = GymV21StepAPIWrapper(env)
    # env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    # Ensure actions passed to Box2D are float32
    return env
