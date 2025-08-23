from asyncio import InvalidStateError

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mmap
import win32event


# 28 bytes shared region: [proceed flag(1 byte)] [action(3 bytes)] [reward(1 byte)] [terminated(1 byte)] [empty padding(2 bytes)] [state(20 bytes (5x floats))]
FILE_SIZE = 28

# Create anonymous memory-mapped file with a local name
mm = mmap.mmap(-1, FILE_SIZE, tagname="Local\\ATCRLSharedMem")

# Named events for synchronization
reset_sim = win32event.CreateEvent(None, False, False, "Local\\ATCRLResetEvent")
action_ready = win32event.CreateEvent(None, False, False, "Local\\ATCRLActionReadyEvent")
action_done = win32event.CreateEvent(None, False, False, "Local\\ATCRLActionDoneEvent")


class TC2Env(gym.Env):
    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()

        # Actions[0] = [steps of 5 degree from 0-355]
        # Actions[1] = [steps of 1000 feet from min to max altitude - 2000 to FL150 for Singapore]
        # Actions[2] = [steps of 10 knots from 160 to 250 knots (for now)]
        self.action_space = spaces.MultiDiscrete([360 // 5, 14, 10])

        # [x, y, alt, gs, track] normalized
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        self.state_multiplier = np.array([4000, 4000, 15000 - 2000, 600, 360])
        self.state_adder = np.array([-2000, -2000, 2000, 0, 0])

        self.steps = 0
        self.max_steps = 400
        self.render_mode = render_mode

    def normalize_sim_state(self, sim_state) -> np.ndarray:
        return (sim_state - self.state_adder) / self.state_multiplier

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Send reset signal to simulator
        win32event.SetEvent(reset_sim)
        # Wait for simulator to signal ready for next action
        win32event.WaitForSingleObject(action_ready, win32event.INFINITE)

        # TODO Get state from shared memory
        raise NotImplementedError()

        info = {}
        return obs, info

    def step(self, action):
        # Validate that simulator is ready to accept action (proceed flag)
        mm.seek(0)
        proceed_flag = mm.read_byte()
        if proceed_flag != 1:
            raise InvalidStateError("Proceed flag must be 1")
        mm.write_byte(0) # Reset flag

        # TODO Write action to shared memory

        # Wait till simulator finished simulating 300 frames (action_ready event)
        # TODO Then read state, reward, terminated, truncated from shared memory
        win32event.WaitForSingleObject(action_ready, win32event.INFINITE)
        raise NotImplementedError()

        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass


while True:
    flag_val = int(input("Input an integer to set: "))
    mm.seek(0)
    mm.write(flag_val.to_bytes())
