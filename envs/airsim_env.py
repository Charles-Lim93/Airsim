import numpy as np
import airsim

import gym
from gym import spaces


class AirSimEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, image_shape):
        self.observation_space = spaces.Box(0, 255, shape=image_shape, dtype=np.uint8)
        self.viewer = None

    def __del__(self):
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _compute_reward(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def render(self):
        return self._get_obs()

class AirSimContinuousEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, image_shape,attitude_shape):
        # self.observation_space = spaces.Box(low = 0, high = 255, shape=image_shape, dtype=np.uint8)
        self.observation_space = spaces.Dict(
            {
                # Sequential Image Data
                "Image": spaces.Box(low = 0, high = 255, shape=image_shape, dtype=np.uint8),
                # Sequential Attitude Data
                "Linear velocity": spaces.Box(low = -5, high = 5, shape=attitude_shape, dtype=np.float64),  
            }
        )
# /        self.__dict__(spaces.Dict(self.observation_space))
        self.viewer = None

    def __del__(self):
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _compute_reward(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def render(self):
        return self._get_obs()
