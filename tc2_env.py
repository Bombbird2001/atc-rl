import gymnasium as gym
import numpy as np
import os
import subprocess

from game_bridge import GameBridge
from gymnasium import spaces
from enum import Enum
from typing import List, Optional


SIMULATOR_JAR = os.getenv("SIMULATOR_JAR")


class TC2Env(gym.Env):
    def __init__(
            self, is_eval=False, render_mode=None, reset_print_period=1, instance_suffix="",
            init_sim=True, max_steps=400
    ):
        super().__init__()

        self.sim_bridge = GameBridge.get_bridge_for_platform(instance_suffix=instance_suffix)

        self.instance_name = f"env{instance_suffix}"

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
        self.max_steps = max_steps
        self.terminated_count = 0
        self.render_mode = render_mode

        print(f"[{self.instance_name}] Environment initialized")

        if init_sim:
            print(f"[{self.instance_name}] Starting simulator")
            self.sim_process = subprocess.Popen(f"java -jar \"{SIMULATOR_JAR}\" {instance_suffix}", shell=True)

        # At launch, send a single reset signal since reset may be called after simulator starts in multiple environments mode
        self.sim_bridge.signal_reset_sim()

    def normalize_sim_state(self, sim_state) -> np.ndarray:
        return (sim_state - self.state_adder) / self.state_multiplier

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Send reset signal to simulator
        self.sim_bridge.signal_reset_sim()
        # Wait for simulator to signal ready for next action
        if self.episode % self.reset_print_period == 0:
            if self.episode > 0:
                print(f"[{self.instance_name}] {self.terminated_count} / {self.reset_print_period} episodes terminated before max_steps")
            print(f"[{self.instance_name}] Waiting for action ready after reset: episode {self.episode}")
            self.terminated_count = 0
        self.sim_bridge.wait_action_ready()

        # Get state from shared memory
        values = self.sim_bridge.get_aircraft_state()
        obs = self.normalize_sim_state(np.array(values, dtype=np.float32))
        # print(obs)

        info = {}
        self.episode += 1
        self.steps = 0
        return obs, info

    def step(self, action):
        # Validate that simulator is ready to accept action (proceed flag)
        values = self.sim_bridge.get_total_state()
        proceed_flag = values[0]
        if proceed_flag != 1:
            raise ValueError(f"[{self.instance_name}] Proceed flag must be 1")

        # Write action to shared memory and signal
        self.sim_bridge.write_actions(action[0], action[1], action[2])
        # print(action)

        # Set the reset request flag before signalling action done
        # The next time the game loop finishes simulating 300 frames, it will stop the update till reset() is called here
        self.steps += 1
        truncated = self.max_steps is not None and self.steps >= self.max_steps
        if truncated and not self.is_eval:
            # print(f"Truncating={truncated}")
            self.sim_bridge.signal_reset_after_step()

        # print(int(time.time() * 1000), "Signalled action done")
        self.sim_bridge.signal_action_done()

        # print("Waiting for action ready")

        # Wait till simulator finished simulating 300 frames (action_ready event)
        self.sim_bridge.wait_action_ready()

        # Read state, reward, terminated, truncated from shared memory
        values = self.sim_bridge.get_total_state()
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
        self.sim_bridge.close()


class MCTSPartialState(Enum):
    HDG_SELECTED = 0
    HDG_ALT_SELECTED = 1
    ALL_SELECTED = 2


class MCTSState:
    def __init__(
            self, backing_env: TC2Env, state: np.ndarray, terminated: bool, terminal_reward: float,
            state_type: MCTSPartialState, hdg_action: Optional[int], alt_action: Optional[int], spd_action: Optional[int]
    ):
        self.backing_env = backing_env
        self.state = state
        self.terminated = terminated
        self.terminal_reward = terminal_reward
        self.state_type = state_type

        self.hdg_action = hdg_action
        self.alt_action = alt_action
        self.spd_action = spd_action

        self.possible_actions_hdg = list(range(backing_env.action_space.n_vec[0]))
        self.possible_actions_alt = list(range(backing_env.action_space.n_vec[1]))
        self.possible_actions_spd = list(range(backing_env.action_space.n_vec[2]))

    def getPossibleActions(self) -> List[int]:
        if self.state_type == MCTSPartialState.HDG_SELECTED:
            return self.possible_actions_alt
        elif self.state_type == MCTSPartialState.HDG_ALT_SELECTED:
            return self.possible_actions_spd
        return self.possible_actions_hdg

    def takeAction(self, action: int):
        if self.state_type == MCTSPartialState.HDG_SELECTED:
            return MCTSState(
                self.backing_env, self.state, self.terminated, self.terminal_reward, MCTSPartialState.HDG_ALT_SELECTED,
                self.hdg_action, self.alt_action, None
            )
        if self.state_type == MCTSPartialState.HDG_ALT_SELECTED:
            combined_actions = (self.hdg_action, self.alt_action, action)
            obs, reward, terminated, _, _ = self.backing_env.step(combined_actions)
            return MCTSState(
                self.backing_env, obs, terminated, terminal_reward, MCTSPartialState.ALL_SELECTED,
                None, None, None
            )

        return MCTSState(
            self.backing_env, self.state, self.terminated, self.terminal_reward, MCTSPartialState.HDG_SELECTED,
            self.hdg_action, None, None
        )

    def isTerminal(self):
        return self.terminated

    def getReward(self):
        return self.terminal_reward

    @classmethod
    def getRootState(cls, backing_env: TC2Env, state: np.ndarray):
        return MCTSState(
            backing_env, state, False, 1e20, MCTSPartialState.ALL_SELECTED,
            None, None, None
        )


def make_env(env_id: int, processes: List, auto_init_sim: bool):
    backing_env = TC2Env(render_mode="human", reset_print_period=50, instance_suffix=str(env_id), init_sim=auto_init_sim)
    if auto_init_sim:
        processes.append(backing_env.sim_process)
    return backing_env
