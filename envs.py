"""
Shared environment wrappers.

ContinuousCartPoleWrapper
    Adapts CartPole-v1 for SAC / other continuous-action algorithms by
    exposing a Box([-1.0], [1.0]) action space.  The underlying dynamics
    and reward function are unchanged: negative action -> push left (0),
    non-negative action -> push right (1).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ContinuousCartPoleWrapper(gym.Wrapper):
    """
    Wraps CartPole-v1 with a 1-D continuous action space in [-1.0, 1.0].

    Mapping:  a < 0  ->  discrete 0  (push left)
              a >= 0 ->  discrete 1  (push right)

    Observation space and reward function are preserved exactly.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def step(self, action):
        discrete = int(np.asarray(action).flat[0] >= 0.0)
        return self.env.step(discrete)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def make_continuous_cartpole(seed: int | None = None) -> gym.Env:
    """Convenience factory: wrapped CartPole-v1 ready for continuous-action RL."""
    env = gym.make("CartPole-v1")
    env = ContinuousCartPoleWrapper(env)
    if seed is not None:
        env.reset(seed=seed)
    return env
