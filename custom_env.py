import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CounterEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, target=10, render_mode=None):
        super().__init__()

        # Actions: 0 = decrement, 1 = increment
        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Box(
            low=np.array([0]), high=np.array([target]), dtype=np.int32
        )

        self.state = 0
        self.target = target
        self.steps = 0
        self.max_steps = 20
        self.render_mode = render_mode

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        self.steps = 0
        obs = np.array([self.state], dtype=np.int32)
        info = {}
        return obs, info

    def step(self, action):
        if action == 1:
            self.state += 1
        else:
            self.state -= 1
        self.steps += 1

        # Reward: negative distance to target
        reward = -abs(self.target - self.state)

        # Termination condition
        terminated = self.state == self.target
        truncated = self.steps >= self.max_steps

        obs = np.array([self.state], dtype=np.int32)
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"Step: {self.steps} | State: {self.state}")
        elif self.render_mode == "rgb_array":
            # Dummy image
            return np.zeros((200, 200, 3), dtype=np.uint8)

    def close(self):
        pass
