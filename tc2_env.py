from asyncio import InvalidStateError

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mmap
import random
import struct
import win32event


# 32 bytes shared region: [proceed flag(1 byte)] [action(3 bytes)] [reward(4 bytes (1 float))] [terminated(1 byte)] [empty padding(2 bytes)] [state(20 bytes (5x floats))]
FILE_SIZE = 32
STRUCT_FORMAT = "bbbbf?xxxfffff"

# Create anonymous memory-mapped file with a local name
mm = mmap.mmap(-1, FILE_SIZE, tagname="Local\\ATCRLSharedMem")

# Named events for synchronization
reset_sim = win32event.CreateEvent(None, False, False, "Local\\ATCRLResetEvent")
action_ready = win32event.CreateEvent(None, False, False, "Local\\ATCRLActionReadyEvent")
action_done = win32event.CreateEvent(None, False, False, "Local\\ATCRLActionDoneEvent")
reset_after_step = win32event.CreateEvent(None, False, False, "Local\\ATCRLResetAfterEvent")


class TC2Env(gym.Env):
    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, is_eval=False, render_mode=None):
        super().__init__()

        self.is_eval = is_eval

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

        self.episode = 0
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
        print(f"Waiting for action ready after reset: episode {self.episode}")
        win32event.WaitForSingleObject(action_ready, win32event.INFINITE)

        # Get state from shared memory
        mm.seek(12)
        bytes_read = mm.read(20)
        values = struct.unpack("fffff", bytes_read)
        obs = self.normalize_sim_state(np.array(values))
        # print(obs)

        info = {}
        self.episode += 1
        return obs, info

    def step(self, action):
        # Validate that simulator is ready to accept action (proceed flag)
        mm.seek(0)
        values = struct.unpack(STRUCT_FORMAT, mm.read(FILE_SIZE))
        # print(int(time.time() * 1000), values)
        proceed_flag = values[0]
        if proceed_flag != 1:
            raise InvalidStateError("Proceed flag must be 1")

        # Write action to shared memory and signal
        mm.seek(0)
        mm.write(struct.pack("bbbb", 1, action[0], action[1], action[2]))

        # Set the reset request flag before signalling action done
        # The next time the game loop finishes simulating 300 frames, it will stop the update till reset() is called here
        self.steps += 1
        truncated = self.steps >= self.max_steps
        if (truncated and not self.is_eval) or values[5]:
            # print(f"Truncating={truncated}, Terminating={values[5]}")
            win32event.SetEvent(reset_after_step)

        # print(int(time.time() * 1000), "Signalled action done")
        win32event.SetEvent(action_done)

        # print("Waiting for action ready")

        # Wait till simulator finished simulating 300 frames (action_ready event)
        win32event.WaitForSingleObject(action_ready, win32event.INFINITE)
        # print(int(time.time() * 1000), "Action ready")

        # Read state, reward, terminated, truncated from shared memory
        mm.seek(0)
        values = struct.unpack(STRUCT_FORMAT, mm.read(FILE_SIZE))
        obs = self.normalize_sim_state(np.array(values[6:11]))
        reward = values[4]
        terminated = values[5]

        info = {}

        # print(obs, reward, terminated, truncated)
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    env = TC2Env()
    env.reset()
    while True:
        env.step(np.array([random.randint(0, 360 // 5 - 1), random.randint(0, 13), random.randint(0, 9)]))
