import gymnasium as gym
import numpy as np

from typing import Any, Callable
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, WrapperObsType

from typing import Optional, Union, Tuple



import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

class ColorTransformObservation(ObservationWrapper):
    def __init__(self, env, color="green"):
        """
        Observation wrapper to apply a color tint over image observations.

        Args:
            env: The environment to wrap.
            color (str): The tint color to apply. Options: "green", "red", "blue".
        """
        super().__init__(env)
        assert color in ["standard", "green", "red", "blue"], "Color must be 'green', 'red', or 'blue'."
        self.color = color
        
        # Ensure the observation space is an image space
        assert isinstance(self.observation_space, Box), "Observation space must be a Box space."
        assert len(self.observation_space.shape) == 3, "Observations must be 3D images (H, W, C)."
        assert self.observation_space.shape[2] == 3, "Observations must have 3 channels (RGB)."
        
        # Keep observation space unchanged
        self.observation_space = env.observation_space
    
    def observation(self, obs):
        """
        Modify the observation by applying the color tint.

        Args:
            obs (np.ndarray): Original observation.

        Returns:
            np.ndarray: Observation with the color tint applied.
        """
        # Create a tint mask based on the selected color
        tint = np.zeros_like(obs, dtype=np.float32)
        if self.color == "red":
            tint[:, :, 0] = 1.0
        elif self.color == "green":
            tint[:, :, 1] = 1.0
        elif self.color == "blue":
            tint[:, :, 2] = 1.0
        elif self.color == "standard":
            # No tint
            return obs
        
        # Apply the tint with a blending factor
        blend_factor = 0.2
        obs_tinted = (1 - blend_factor) * obs + blend_factor * tint * 255
        return np.clip(obs_tinted, 0, 255).astype(np.uint8)



class TransformObservation(
    gym.ObservationWrapper[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Applies a function to the ``observation`` received from the environment's :meth:`Env.reset` and :meth:`Env.step` that is passed back to the user.

    The function :attr:`func` will be applied to all observations.
    If the observations from :attr:`func` are outside the bounds of the ``env``'s observation space, provide an updated :attr:`observation_space`.

    A vector version of the wrapper exists :class:`gymnasium.wrappers.vector.TransformObservation`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import TransformObservation
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> env = gym.make("CartPole-v1")
        >>> env.reset(seed=42)
        (array([ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ], dtype=float32), {})
        >>> env = gym.make("CartPole-v1")
        >>> env = TransformObservation(env, lambda obs: obs + 0.1 * np.random.random(obs.shape), env.observation_space)
        >>> env.reset(seed=42)
        (array([0.08227695, 0.06540678, 0.09613613, 0.07422512]), {})

    Change logs:
     * v0.15.4 - Initially added
     * v1.0.0 - Add requirement of ``observation_space``
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        func: Callable[[ObsType], Any],
        # observation_space: gym.Space[WrapperObsType] | None,
        observation_space: Optional[gym.Space[WrapperObsType]] = None,
    ):
        """Constructor for the transform observation wrapper.

        Args:
            env: The environment to wrap
            func: A function that will transform an observation. If this transformed observation is outside the observation space of ``env.observation_space`` then provide an `observation_space`.
            observation_space: The observation spaces of the wrapper, if None, then it is assumed the same as ``env.observation_space``.
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, func=func, observation_space=observation_space
        )
        gym.ObservationWrapper.__init__(self, env)

        if observation_space is not None:
            self.observation_space = observation_space

        self.func = func

    def observation(self, observation: ObsType) -> Any:
        """Apply function to the observation."""
        return self.func(observation)


class ReshapeObservation(
    TransformObservation[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Reshapes Array based observations to a specified shape.

    A vector version of the wrapper exists :class:`gymnasium.wrappers.vector.RescaleObservation`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import ReshapeObservation
        >>> env = gym.make("CarRacing-v2")
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> reshape_env = ReshapeObservation(env, (24, 4, 96, 1, 3))
        >>> reshape_env.observation_space.shape
        (24, 4, 96, 1, 3)

    Change logs:
     * v1.0.0 - Initially added
    """

    # shape: int | tuple[int, ...]):
    def __init__(
        self, env: gym.Env[ObsType, ActType], shape: Union[int, Tuple[int, ...]]
    ):
        """Constructor for env with ``Box`` observation space that has a shape product equal to the new shape product.

        Args:
            env: The environment to wrap
            shape: The reshaped observation space
        """
        assert isinstance(env.observation_space, spaces.Box)
        assert np.prod(shape) == np.prod(env.observation_space.shape)

        assert isinstance(shape, tuple)
        assert all(np.issubdtype(type(elem), np.integer) for elem in shape)
        assert all(x > 0 or x == -1 for x in shape)

        new_observation_space = spaces.Box(
            low=np.reshape(np.ravel(env.observation_space.low), shape),
            high=np.reshape(np.ravel(env.observation_space.high), shape),
            shape=shape,
            dtype=env.observation_space.dtype,
        )
        self.shape = shape
        gym.utils.RecordConstructorArgs.__init__(self, shape=shape)
        TransformObservation.__init__(
            self,
            env=env,
            # func=lambda obs: np.reshape(obs, shape),
            func=lambda obs: np.transpose(obs, (2,0,1)),
            observation_space=new_observation_space,
        )


class RescaleObservation(
    TransformObservation[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Normalize the observation from [0, 255] to be in the range [0, 1].

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import ReshapeObservation
        >>> env = gym.make("CarRacing-v2")
        >>> env.observation_space.min()
        0
        >>> env.observation_space.max()
        255
        >>> normalize_env = RescaleObservation(env)
        >>> normalize_env.observation_space.min()
        0
        >>> normalize_env.observation_space.max()
        1


    Change logs:
     * v1.0.0 - Initially added
    """

    # shape: int | tuple[int, ...]):
    def __init__(self, env: gym.Env[ObsType, ActType], rescale_value=255.0):
        """Constructor for env with ``Box`` observation space that has a shape product equal to the new shape product.

        Args:
            env: The environment to wrap
            shape: The reshaped observation space
        """
        assert isinstance(env.observation_space, spaces.Box)
        # assert np.prod(shape) == np.prod(env.observation_space.shape)

        # assert isinstance(shape, tuple)
        # assert all(np.issubdtype(type(elem), np.integer) for elem in shape)
        # assert all(x > 0 or x == -1 for x in shape)

        # new_observation_space = spaces.Box(
        #     low=np.zeros(shape, dtype=env.observation_space.dtype),
        #     high=np.ones(shape, dtype=env.observation_space.dtype),
        #     dtype=env.observation_space.dtype,
        # )
        # self.shape = shape

        new_observation_space = spaces.Box(
            low=env.observation_space.low / rescale_value,
            high=env.observation_space.high / rescale_value,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )
        # gym.utils.RecordConstructorArgs.__init__(self, shape=shape)
        TransformObservation.__init__(
            self,
            env=env,
            func=lambda obs: obs / rescale_value,
            observation_space=new_observation_space,
        )


# class FilterFromDict(gym.ObservationWrapper):
#     from typing import List
#     # Minigrid: Dict('direction': Discrete(4), 'image': Box(0, 255, (7, 7, 3), uint8), 'mission': MissionSpace(<function EmptyEnv._gen_mission at 0x12dfeb4c0>, None))
#     """
#     Filter the observation from the dict to return only values for the specified key
#     """
#     def __init__(self, env=None, key:str='image'):
#         super(FilterFromDict, self).__init__(env)
#         self.env = env
#         self.key = key
#         self.observation_space = env.observation_space.spaces[self.key]

#         TransformObservation.__init__(
#             self,
#             env=env,
#             func=lambda obs: obs[self.key],
#             observation_space=self.observation_space,
#         )


class FilterFromDict(gym.ObservationWrapper):
    from typing import List

    # Minigrid: Dict('direction': Discrete(4), 'image': Box(0, 255, (7, 7, 3), uint8), 'mission': MissionSpace(<function EmptyEnv._gen_mission at 0x12dfeb4c0>, None))
    """
    Filter the observation from the dict to only return values for the specified key
    """

    def __init__(self, env=None, key: str = None):
        super(FilterFromDict, self).__init__(env)
        self.env = env
        if key is None:
            print("keys is None, trying to use 'image' as default")
            key = "image"
        self.key = key
        self.observation_space = env.observation_space.spaces[self.key]

    def observation(self, observation):
        observation = observation["image"]  # print(observation['image'])
        return observation  # [0][self.key]


class RepeatAction(gym.Wrapper):
    """
    Repeat the same action a certain number of times.
    Often, in a game, a couple of sequential frames do not differ that much from each other, so repeat the chosen action.
    """

    def __init__(self, env=None, repeat=4, clip_rewards=False, fire_first=False):
        """
        :param env:
        :param repeat:
        :param fire_first: If the rl_agent need to start the game by pressing "fire" button
        """
        # fire_first: in certain envs the rl_agent have to fire to start the env, as in pong
        # the rl_agent can figure it out alone sometimes
        super(RepeatAction, self).__init__(env)
        self.env = env
        self.repeat = repeat
        self.clip_rewards = clip_rewards
        self.shape = env.observation_space.low.shape
        self.fire_first = fire_first

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, truncated, terminated, info = self.env.step(action)
            if self.clip_rewards:
                # clip the reward in -1, 1, then take first element (we need the scalar, not an array)
                reward = np.sign(reward)  # np.clip(np.array([reward]), -1, 1)[0]
            total_reward += reward

            done = truncated or terminated
            if done:
                break
        return obs, total_reward, truncated, terminated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.fire_first:
            # get_action_meanings returns a list of strings (['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'])
            fire_act_idx = self.env.unwrapped.get_action_meanings().index("FIRE")
            obs, _, _, _ = self.env.step(fire_act_idx)
        return obs, info


############### old normalization methods


# class RepeatAction(gym.Wrapper):
#     """
#     Repeat the same action a certain number of times.
#     Often, in a game, a couple of sequential frames do not differ that much from each other, so repeat the chosen action.
#     """

#     def __init__(self, env=None, repeat=4, clip_rewards=False, fire_first=False):
#         """
#         :param env:
#         :param repeat:
#         :param fire_first: If the rl_agent need to start the game by pressing "fire" button
#         """
#         # fire_first: in certain envs the rl_agent have to fire to start the env, as in pong
#         # the rl_agent can figure it out alone sometimes
#         super(RepeatAction, self).__init__(env)
#         self.env = env
#         self.repeat = repeat
#         self.clip_rewards = clip_rewards
#         self.shape = env.observation_space.low.shape
#         self.fire_first = fire_first

#     def step(self, action):
#         total_reward = 0.0
#         done = False
#         for i in range(self.repeat):
#             obs, reward, done, info = self.env.step(action)
#             if self.clip_rewards:
#                 # clip the reward in -1, 1, then take first element (we need the scalar, not an array)
#                 reward = np.sign(reward) # np.clip(np.array([reward]), -1, 1)[0]

#             total_reward += reward
#             if done:
#                 break
#         return obs, total_reward, done, info

#     def reset(self):
#         obs = self.env.reset()
#         if self.fire_first:
#             # get_action_meanings returns a list of strings (['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'])
#             fire_act_idx = self.env.unwrapped.get_action_meanings().index("FIRE")
#             obs, _, _, _ = self.env.step(fire_act_idx)
#         return obs


# # https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py


# class EpisodicLifeEnv(gym.Wrapper):
#     def __init__(self, env):
#         """Make end-of-life == end-of-episode, but only reset on true game over.
#         Done by DeepMind for the DQN and co. since it helps value estimation.
#         """
#         gym.Wrapper.__init__(self, env)
#         self.lives = 0
#         self.was_real_done = True

#     def step(self, action):
#         obs, reward, done, info = self.env.step(action)
#         self.was_real_done = done
#         # check current lives, make loss of life terminal,
#         # then update lives to handle bonus lives
#         lives = self.env.unwrapped.ale.lives()
#         if lives < self.lives and lives > 0:
#             # for Qbert sometimes we stay in lives == 0 condition for a few frames
#             # so it's important to keep lives > 0, so that we only reset once
#             # the environment advertises done.
#             done = True
#         self.lives = lives
#         return obs, reward, done, info

#     def reset(self):
#         """Reset only when lives are exhausted.
#         This way all states are still reachable even though lives are episodic,
#         and the learner need not know about any of this behind-the-scenes.
#         """
#         if self.was_real_done:
#             obs = self.env.reset()
#         else:
#             # no-op step to advance from terminal/lost life state
#             obs, _, _, _ = self.env.step(0)
#         self.lives = self.env.unwrapped.ale.lives()
#         return obs


# class PreprocessFrame(gym.ObservationWrapper):
#     """
#     Reduce frame size and convert to grayscale with values in range [0, 1]
#     """

#     def __init__(self, shape, env=None):
#         super(PreprocessFrame, self).__init__(env)
#         self.env = env
#         # pytorch expects channel firsts
#         self.shape = (shape[2], shape[0], shape[1])
#         # normalize values in 0, 1 and in grayscale
#         self.observation_space = gym.spaces.Box(
#             low=0.0, high=1.0, shape=self.shape, dtype=np.float32
#         )

#     def observation(self, observation):
#         new_frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
#         # self.shape[1:] because it is 42x42 (w, h) (shape[0] indicates the num_channels)
#         resized_screen = cv2.resize(
#             new_frame, self.shape[1:], interpolation=cv2.INTER_AREA
#         )
#         # uint8 because we have obs as ints for 0 to 255 (might not be needed), since we're converting to float
#         new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
#         new_obs = new_obs / 255.0
#         return new_obs

# class PreprocessFrameRGB(gym.ObservationWrapper):
#     """
#     Reduce frame size and convert to grayscale with values in range [0, 1]
#     """

#     def __init__(self, shape, env=None):
#         super(PreprocessFrameRGB, self).__init__(env)
#         self.env = env
#         # pytorch expects channel firsts
#         self.shape = (shape[2], shape[0], shape[1])
#         # normalize values in 0, 1 and in grayscale
#         self.observation_space = gym.spaces.Box(
#             low=env.observation_space.low[0, 0, 0], high=env.observation_space.high[255, 255, 255], shape=self.shape, dtype=np.float32
#             # low=0.0, high=1.0, shape=self.shape, dtype=np.float32
#         )

#     def observation(self, observation):
#         # new_frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
#         # self.shape[1:] because it is 42x42 (w, h) (shape[0] indicates the num_channels)
#         # print(observation)
#         resized_screen = cv2.resize(
#             observation, self.shape[1:], interpolation=cv2.INTER_AREA
#         )
#         # uint8 because we have obs as ints for 0 to 255 (might not be needed), since we're converting to float
#         new_obs = np.array(resized_screen, dtype=np.uint8).transpose((2, 0, 1))
#         new_obs = new_obs# / 255.0
#         return new_obs


# class NormalizeFrames(gym.ObservationWrapper):
#     """
#     Normalize frames to be in range [0, 1]
#     """

#     def __init__(self, env=None):
#         super(NormalizeFrames, self).__init__(env)
#         self.env = env
#         self.observation_space = gym.spaces.Box(
#             low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32
#         )

#     def observation(self, observation):
#         return np.array(observation).astype(np.float32) / 255.0

# class FilterFromDict(gym.ObservationWrapper):
#     from typing import List
#     # Minigrid: Dict('direction': Discrete(4), 'image': Box(0, 255, (7, 7, 3), uint8), 'mission': MissionSpace(<function EmptyEnv._gen_mission at 0x12dfeb4c0>, None))
#     """
#     Filter the observation from the dict to only return values for the specified key
#     """
#     def __init__(self, env=None, key:str=None):
#         super(FilterFromDict, self).__init__(env)
#         self.env = env
#         if key is None:
#             print('keys is None, trying to use \'image\' as default')
#             key = 'image'
#         self.key = key
#         self.observation_space = env.observation_space.spaces[self.key]
#         print(self.observation_space)

#     def observation(self, observation):
#         return observation[0][self.key]

# class StackFrames(gym.ObservationWrapper):
#     """
#     Stack a number of frames to give some directional hints to the rl_agent (e.g. in pong, using a single frame,
#     we would not be able to tell the direction the ball is going).
#     """

#     def __init__(self, env, repeat):
#         super(StackFrames, self).__init__(env)
#         self.repeat = repeat
#         # (repeat, w, h)
#         self.observation_space = gym.spaces.Box(
#             env.observation_space.low.repeat(repeat, axis=0),
#             env.observation_space.high.repeat(repeat, axis=0),
#             dtype=np.float32,
#         )
#         self.stack = collections.deque(maxlen=repeat)

#     def reset(self):
#         self.stack.clear()
#         obs = self.env.reset()
#         for i in range(self.repeat):
#             self.stack.append(obs)
#         return np.array(self.stack).reshape(self.observation_space.low.shape)

#     def observation(self, observation):
#         self.stack.append(observation)
#         return np.array(self.stack).reshape(self.observation_space.low.shape)


# def make_custom_env(
#     env, shape=(84, 84, 1), repeat=4, clip_rewards=False, fire_first=False
# ):
#     env = RepeatAction(env, repeat, fire_first, clip_rewards)
#     env = PreprocessFrame(shape, env)
#     env = StackFrames(env, repeat)
#     return env


# class MaxStepsWrapper(gym.Wrapper):
#     def __init__(self, env, max_steps):
#         super(MaxStepsWrapper, self).__init__(env)
#         self.max_steps = max_steps
#         self.step_count = 0

#     def step(self, action):
#         obs, reward, done, info = self.env.step(action)
#         self.step_count += 1
#         if self.step_count >= self.max_steps:
#             done = True
#         return obs, reward, done, info

#     def reset(self):
#         self.step_count = 0
#         return self.env.reset()


# def make_env(
#     env,
#     shape=(84, 84, 1),
#     repeat=4,
#     clip_rewards=False,
#     fire_first=False,
#     episodic_life=False,
#     render_mode="rgb_array",
#     deterministic_env=False,
#     seed=0,
#     max_steps=0,
# ):
#     # env = gym.make(env_name, render_mode)
#     env = RepeatAction(env, repeat, clip_rewards, fire_first)
#     if episodic_life:
#         env = EpisodicLifeEnv(env)
#     if shape[2] == 3:
#         env = PreprocessFrameRGB(shape, env)
#     else:
#         env = PreprocessFrame(shape, env)
#     if repeat > 1:
#         env = StackFrames(env, repeat)
#     if max_steps > 0:
#         env = MaxStepsWrapper(env, max_steps)
#     if deterministic_env:
#         env.set_deterministic(seed)
#     env.seed(seed)
#     env.action_space.seed(seed)
#     env.observation_space.seed(seed)
#     return env


# def make_custom_env_no_stack(
#     env, shape=(84, 84, 1), repeat=4, clip_rewards=False, fire_first=False
# ):
#     env = RepeatAction(env, repeat, clip_rewards, fire_first)
#     env = PreprocessFrame(shape, env)
#     return env

#     # def make_env_clearRL(env, seed, idx, capture_video, run_name):
#     #     # def thunk():
#     #         # env = make_custom_env(env, shape=(84, 84, 1), repeat=4, clip_rewards=False, fire_first=False)
#     #     # env = make_env(env, shape=(84, 84, 1), repeat=4, clip_rewards=False, fire_first=False)
#     #     env = gym.wrappers.RecordEpisodeStatistics(env)
#     #     if capture_video:
#     #         if idx == 0:
#     #             env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#     #     # env = NoopResetEnv(env, noop_max=30)
#     #     # env = MaxAndSkipEnv(env, skip=4)
#     #     # env = EpisodicLifeEnv(env)
#     #     # if "FIRE" in env.unwrapped.get_action_meanings():
#     #     #     env = FireResetEnv(env)
#     #     # env = ClipRewardEnv(env)
#     #     if args.deterministic_env:
#     #         env.set_deterministic(seed)
#     #     env.seed(seed)
#     #     env.action_space.seed(seed)
#     #     env.observation_space.seed(seed)
#     # return env

#     return env  # thunk


# def make_env_cleanrl(env, seed, idx, capture_video, run_name):
#     def thunk(env=env):
#         # env = gym.make(env_id)
#         # env = CarRacing(continuous=False, background='red')
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         if capture_video:
#             if idx == 0:
#                 env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         # env = NoopResetEnv(env, noop_max=30)
#         # env = MaxAndSkipEnv(env, skip=4)
#         # env = EpisodicLifeEnv(env)
#         # if "FIRE" in env.unwrapped.get_action_meanings():
#         #     env = FireResetEnv(env)
#         # env = ClipRewardEnv(env)
#         # env = gym.wrappers.ResizeObservation(env, (84, 84))
#         env = PreprocessFrameRGB((84, 84, 3), env) # (3, 84, 84)
#         # env = gym.wrappers.GrayScaleObservation(env)
#         # env = gym.wrappers.FrameStack(env, 4) #(4, 3, 84, 84)
#         env.seed(seed)
#         env.action_space.seed(seed)
#         env.observation_space.seed(seed)
#         return env

#     return thunk


'''
from PIL import Image
import torch
class PreprocessFrameRGBTimm(gym.ObservationWrapper):
    """
    Reduce frame size and convert to grayscale with values in range [0, 1]
    """

    def __init__(self, shape, transform, env=None):
        super().__init__(env)
        self.transform = transform
        self.env = env
        # pytorch expects channel firsts
        # observation = env.reset()
        # observation = Image.fromarray(observation).convert('RGB')
        # observation = self.transform(observation)
        # self.shape = observation.shape
        # print(observation.shape, observation.dtype, observation.min(), observation.max())
        self.shape = (3, 384, 384)
        # normalize values in 0, 1 and in grayscale
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32
        )


    def observation(self, observation):
        # new_frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        # self.shape[1:] because it is 42x42 (w, h) (shape[0] indicates the num_channels)
        # print('whatsup friend')
        # print(observation.shape)

        # print(self.transform)
        observation = Image.fromarray(observation).convert('RGB')
        # print(self.transform)
        # observation = self.transform(observation)
        # apply transformations singularly
        # Resize(size=384, interpolation=bicubic, max_size=None, antialias=None)
        observation = self.transform.transforms[0](observation)
        # CenterCrop(size=(384, 384))
        observation = self.transform.transforms[1](observation)
        # # ToTensor()
        # observation = self.transform.transforms[2](observation)
        # # normalize the numpy array
        # Normalize(mean=tensor([0.5000, 0.5000, 0.5000]), std=tensor([0.5000, 0.5000, 0.5000]))
        # observation = self.transform.transforms[3](observation)
        # print(observation.shape)
        # uint8 because we have obs as ints for 0 to 255 (might not be needed), since we're converting to float
        # # uint8 because we have obs as ints for 0 to 255 (might not be needed), since we're converting to float
        # new_obs = np.array(resized_screen, dtype=np.uint8).transpose((2, 0, 1))
        return observation


# from torchvision import transforms as pth_transforms
# class PreprocessFrameDino(gym.ObservationWrapper):
#     """
#     Reduce frame size and convert to grayscale with values in range [0, 1]
#     """

#     def __init__(self, shape, env=None):
#         super(PreprocessFrameDino, self).__init__(env)
#         self.env = env
#         # pytorch expects channel firsts
#         self.shape = (shape[2], shape[0], shape[1])
#         # normalize values in 0, 1 and in grayscale
#         self.observation_space = gym.spaces.Box(
#             low=-3.0, high=3.0, shape=self.shape, dtype=np.float32
#         )

#     def observation(self, observation):
#         val_transform = pth_transforms.Compose([
#         # pth_transforms.Resize(256, interpolation=3),
#         # pth_transforms.CenterCrop(224),
#         pth_transforms.Resize(224, resample=Image.BICUBIC),# Resampling.BICUBIC), # pth_transforms.InterpolationMode.BICUBIC), # use Resampling.BICUBIC
#         pth_transforms.ToTensor(),
#         pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         ])
#         # convert observation to PIL image
#         observation = pth_transforms.ToPILImage()(observation)
#         new_obs = val_transform(observation)
#         new_obs = np.array(new_obs, dtype=np.uint8)
#         return new_obs
'''
