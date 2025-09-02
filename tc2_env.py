import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mmap
import os
import struct
import subprocess
import win32event


# 52 bytes shared region: [proceed flag(1 byte)] [action(3 bytes)] [reward(4 bytes (1 float))] [terminated(1 byte)] [empty padding(2 bytes)] [state(40 bytes (7x floats, 3x ints))]
FILE_SIZE = 52
STRUCT_FORMAT = "bbbbf?xxxfffffffiii"


SIMULATOR_DIR = os.getenv("SIMULATOR_DIR")


class TC2Env(gym.Env):
    def __init__(self, is_eval=False, render_mode=None, reset_print_period=1, instance_suffix="", init_sim=True):
        super().__init__()

        self.instance_name = f"env{instance_suffix}"

        # Create anonymous memory-mapped file with a local name
        self.mm = mmap.mmap(-1, FILE_SIZE, tagname=f"Local\\ATCRLSharedMem{instance_suffix}")

        # Named events for synchronization
        self.reset_sim = win32event.CreateEvent(None, False, False, f"Local\\ATCRLResetEvent{instance_suffix}")
        self.action_ready = win32event.CreateEvent(None, False, False, f"Local\\ATCRLActionReadyEvent{instance_suffix}")
        self.action_done = win32event.CreateEvent(None, False, False, f"Local\\ATCRLActionDoneEvent{instance_suffix}")
        self.reset_after_step = win32event.CreateEvent(None, False, False, f"Local\\ATCRLResetAfterEvent{instance_suffix}")

        self.is_eval = is_eval
        self.reset_print_period = reset_print_period

        # Actions[0] = [steps of 5 degree from 0-355]
        # Actions[1] = [steps of 1000 feet from min to max altitude - 2000 to FL150 for Singapore]
        # Actions[2] = [steps of 10 knots from 160 to 250 knots (for now)]
        self.action_space = spaces.MultiDiscrete([360 // 5, 14, 10])

        # [x, y, alt, gs, track, angular speed, vertical speed, current cleared altitude, current cleared heading, current cleared speed] normalized
        self.observation_space = spaces.Box(
            low=np.repeat(-1.0, 10),
            high=np.repeat(1.0, 10),
            dtype=np.float32
        )

        X_MIN = -2000
        X_MAX = 2000
        Y_MIN = -2000
        Y_MAX = 2000
        ALT_MIN = 0
        ALT_MAX = 36000
        GS_MIN = 0
        GS_MAX = 600
        TRACK_MIN = 0
        TRACK_MAX = 360
        ANG_SPD_MIN = -4
        ANG_SPD_MAX = 4
        VERT_SPD_MIN = -7000
        VERT_SPD_MAX = 7000
        CLEARED_ALT_MIN = 2000
        CLEARED_ALT_MAX = 15000
        CLEARED_HDG_MIN = 0
        CLEARED_HDG_MAX = 360
        CLEARED_SPD_MIN = 160
        CLEARED_SPD_MAX = 250

        self.state_multiplier = np.array([
            (X_MAX - X_MIN) / 2, (Y_MAX - Y_MIN) / 2, (ALT_MAX - ALT_MIN) / 2,
            (GS_MAX - GS_MIN) / 2, (TRACK_MAX - TRACK_MIN) / 2,
            (ANG_SPD_MAX - ANG_SPD_MIN) / 2, (VERT_SPD_MAX - VERT_SPD_MIN) / 2,
            (CLEARED_ALT_MAX - CLEARED_ALT_MIN) / 2,
            (CLEARED_HDG_MAX - CLEARED_HDG_MIN) / 2,
            (CLEARED_SPD_MAX - CLEARED_SPD_MIN) / 2,
        ])
        self.state_adder = np.array([
            (X_MAX + X_MIN) / 2, (Y_MAX + Y_MIN) / 2, (ALT_MAX + ALT_MIN) / 2,
            (GS_MAX + GS_MIN) / 2, (TRACK_MAX + TRACK_MIN) / 2,
            (ANG_SPD_MAX + ANG_SPD_MIN) / 2, (VERT_SPD_MAX + VERT_SPD_MIN) / 2,
            (CLEARED_ALT_MAX + CLEARED_ALT_MIN) / 2,
            (CLEARED_HDG_MAX + CLEARED_HDG_MIN) / 2,
            (CLEARED_SPD_MAX + CLEARED_SPD_MIN) / 2,
        ])

        self.episode = 0
        self.steps = 0
        self.max_steps = 400
        self.terminated_count = 0
        self.render_mode = render_mode

        print(f"[{self.instance_name}] Environment initialized")

        if init_sim:
            print(f"[{self.instance_name}] Starting simulator")
            self.sim_process = subprocess.Popen(f"cd {SIMULATOR_DIR} && gradlew --no-daemon :atcRL:run --args='{instance_suffix}'", shell=True)

    def normalize_sim_state(self, sim_state) -> np.ndarray:
        return (sim_state - self.state_adder) / self.state_multiplier

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Send reset signal to simulator
        win32event.SetEvent(self.reset_sim)
        # Wait for simulator to signal ready for next action
        if self.episode % self.reset_print_period == 0:
            if self.episode > 0:
                print(f"[{self.instance_name}] {self.terminated_count} / {self.reset_print_period} episodes terminated before max_steps")
            print(f"[{self.instance_name}] Waiting for action ready after reset: episode {self.episode}")
            self.terminated_count = 0
        win32event.WaitForSingleObject(self.action_ready, win32event.INFINITE)

        # Get state from shared memory
        self.mm.seek(12)
        bytes_read = self.mm.read(40)
        values = struct.unpack("fffffffiii", bytes_read)
        obs = self.normalize_sim_state(np.array(values, dtype=np.float32))
        # print(obs)

        info = {}
        self.episode += 1
        self.steps = 0
        return obs, info

    def step(self, action):
        # Validate that simulator is ready to accept action (proceed flag)
        self.mm.seek(0)
        values = struct.unpack(STRUCT_FORMAT, self.mm.read(FILE_SIZE))
        proceed_flag = values[0]
        if proceed_flag != 1:
            raise ValueError(f"[{self.instance_name}] Proceed flag must be 1")

        # Write action to shared memory and signal
        self.mm.seek(0)
        self.mm.write(struct.pack("bbbb", 1, action[0], action[1], action[2]))
        # print(action)

        # Set the reset request flag before signalling action done
        # The next time the game loop finishes simulating 300 frames, it will stop the update till reset() is called here
        self.steps += 1
        truncated = self.steps >= self.max_steps
        if truncated and not self.is_eval:
            # print(f"Truncating={truncated}")
            win32event.SetEvent(self.reset_after_step)

        # print(int(time.time() * 1000), "Signalled action done")
        win32event.SetEvent(self.action_done)

        # print("Waiting for action ready")

        # Wait till simulator finished simulating 300 frames (action_ready event)
        win32event.WaitForSingleObject(self.action_ready, win32event.INFINITE)

        # Read state, reward, terminated, truncated from shared memory
        self.mm.seek(0)
        values = struct.unpack(STRUCT_FORMAT, self.mm.read(FILE_SIZE))
        # print(values[6:16])
        obs = self.normalize_sim_state(np.array(values[6:16]))
        reward = values[4]
        terminated = values[5]
        # print(obs)
        if terminated:
            self.terminated_count += 1

        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
