import gymnasium as gym
import numpy as np
import os
import pandas as pd
import platform
import random
import signal
import subprocess
import torch
from common.constants import AIRCRAFT_COUNT, SPD_BIAS, SPD_SCALE_DOWN, \
    TRACK_RATE_SCALE_DOWN, X_Y_SCALE_DOWN, PX_PER_NM, ALT_SCALE_DOWN, \
    ALT_RATE_SCALE_DOWN, ALT_BIAS
from common.data_preprocessing import RECAT_MAPPING
from gymnasium import spaces
from enum import Enum
from rl_algos import RLAlgo, RLAlgos
from sklearn.preprocessing import OneHotEncoder
from typing import List, Optional
from utils.game_bridge import GameBridge


SIMULATOR_JAR = os.getenv("SIMULATOR_JAR")


class TC2Env(gym.Env):
    def __init__(
            self, algo: RLAlgo, ac_type_one_hot_encoder: OneHotEncoder, is_eval=False, render_mode=None, reset_print_period=1, instance_suffix="",
            init_sim=True, max_steps=300
    ):
        super().__init__()

        if init_sim:
            instance_suffix = f"{instance_suffix}_{random.randbytes(3).hex()}"
        self.init_sim = init_sim
        self.sim_bridge = GameBridge.get_bridge_for_platform(instance_suffix=instance_suffix)
        self.signalled_ready = False

        self.instance_name = f"env{instance_suffix}"

        self.is_eval = is_eval
        self.reset_print_period = reset_print_period

        self.action_requires_processing = False
        if algo == RLAlgos.PPO:
            # Actions[0] = [aircraft 0 to 14, or no clearance (value = 0)]
            # Actions[1] = [steps of 5 degrees from 0-359]
            # Actions[2] = [steps of 1000 feet from min to max altitude - 2000 to FL150 for Singapore]
            # Actions[3] = [steps of 10 knots from 160 to 250 knots (for now)]
            self.action_space = spaces.MultiDiscrete([1 + AIRCRAFT_COUNT, 72, 14, 10])
        else:
            # Actions[0] = [continuous 0-360]
            # Actions[1] = [continuous 2000 to FL150 for Singapore]
            # Actions[2] = [continuous 160 to 250 knots (for now)]
            # TODO Currently does not support selection from multiple aircraft
            self.action_space = spaces.Box(
                low=np.repeat(-1.0, 3),
                high=np.repeat(1.0, 3),
                dtype=np.float32,
            )
            self.action_requires_processing = True

        ACT_HDG_MIN = 0
        ACT_HDG_MAX = 360
        ACT_ALT_MIN = 2000
        ACT_ALT_MAX = 15000
        ACT_SPD_MIN = 160
        ACT_SPD_MAX = 250
        # Hdg: [-1, 1] to [0, 1, ..., 359]
        # Alt: [-1, 1] to [0, 1, ..., 13]
        # Spd: [-1, 1] to [0, 1, ..., 9]
        self.action_multiplier = np.array([
            (ACT_HDG_MAX - ACT_HDG_MIN) / 2,
            (ACT_ALT_MAX - ACT_ALT_MIN) / 2000,
            (ACT_SPD_MAX - ACT_SPD_MIN) / 20,
        ])
        self.action_adder = np.array([
            (ACT_HDG_MIN + ACT_HDG_MAX) / 2,
            (ACT_ALT_MIN + ACT_ALT_MAX) / 2000 - 2,
            (ACT_SPD_MIN + ACT_SPD_MAX) / 20 - 16,
        ])

        # [aircraft type, x, y, alt, gs, track, angular speed, vertical speed,
        # current cleared altitude, current cleared heading, current cleared speed, localizer captured] normalized
        # +1 for aircraft masking
        self.OBS_SPACE_DIMENSION = 19
        self.observation_space = spaces.Box(
            low=np.repeat(-1.0, self.OBS_SPACE_DIMENSION * AIRCRAFT_COUNT),
            high=np.repeat(1.0, self.OBS_SPACE_DIMENSION * AIRCRAFT_COUNT),
            dtype=np.float32
        )

        # Remove nuisance missing feature name warning since we're using numpy during inference with no column names
        ac_type_one_hot_encoder.feature_names_in_ = None
        self.ac_type_one_hot_encoder = ac_type_one_hot_encoder

        self.episode = 0
        self.steps = 0
        self.max_steps = max_steps
        self.terminated_count = 0
        self.render_mode = render_mode

        self.action_dist = []

        print(f"[{self.instance_name}] Environment initialized")

        if init_sim:
            print(f"[{self.instance_name}] Starting simulator")
            self.sim_process = subprocess.Popen(f"java -jar \"{SIMULATOR_JAR}\" {instance_suffix}", shell=True)

    def get_observation_from_aircraft_state(self, aircraft_state) -> np.ndarray:
        tmp_state = np.array(aircraft_state).reshape(AIRCRAFT_COUNT, -1)
        # print(tmp_state)
        ac_types = np.array(tmp_state[:,:4], dtype=np.str_)
        ac_types = [[RECAT_MAPPING.get(ac_type, "Unknown")] for ac_type in (ac_types[:,0] + ac_types[:,1] + ac_types[:,2] + ac_types[:,3])]
        ac_type_one_hot = self.ac_type_one_hot_encoder.transform(ac_types).toarray()
        # Map
        # ICAO type, x, y, alt, ias, track, track rate, vertical speed, cleared alt, cleared hdg, cleared IAS, LOC cap, mask
        # to
        # ["ias", "track_rate", "x", "y", "combined_alt", "combined_alt_rate", "track_x", "track_y", "prev_cleared_hdg_x", "prev_cleared_hdg_y",
        # "prev_cleared_alt", "prev_cleared_ias"] + [f"aircraft_type_{j}" for j in range(aircraft_category_count)] + mask
        ac_state = np.array(tmp_state[:,4:], dtype=np.float32)
        # print(ac_state[0])
        combined_ac_state = np.hstack((
            (ac_state[:,[3, 5, 0, 1, 2, 6]] - np.array([SPD_BIAS, 0, 0, 0, ALT_BIAS, 0]))
            / np.array([SPD_SCALE_DOWN, TRACK_RATE_SCALE_DOWN, X_Y_SCALE_DOWN * PX_PER_NM, X_Y_SCALE_DOWN * PX_PER_NM, ALT_SCALE_DOWN, ALT_RATE_SCALE_DOWN]),
            np.sin(np.radians(ac_state[:,[4]])), np.cos(np.radians(ac_state[:,[4]])),
            np.sin(np.radians(ac_state[:,[8]])), np.cos(np.radians(ac_state[:,[8]])),
            (ac_state[:,[7, 9]] - np.array([0, SPD_BIAS])) / np.array([ALT_SCALE_DOWN, SPD_SCALE_DOWN]),
            ac_type_one_hot,
            ac_state[:,[11]]
        ))
        # print(combined_ac_state.shape)
        # TODO Move pre-processing here
        return combined_ac_state.reshape(1, -1)

    def convert_action(self, action) -> np.ndarray:
        return np.rint((action * self.action_multiplier) + self.action_adder).astype(int)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if not self.signalled_ready:
            self.sim_bridge.signal_trainer_initialized()
            self.signalled_ready = True

        # Send reset signal to simulator
        self.sim_bridge.signal_reset_sim()
        # Wait for simulator to signal ready for next action
        if self.episode % self.reset_print_period == 0:
            if self.episode > 0:
                print(f"[{self.instance_name}] {self.terminated_count} / {self.reset_print_period} episodes terminated before max_steps")
            print(f"[{self.instance_name}] Waiting for action ready after reset: episode {self.episode}")
            self.terminated_count = 0

            # Print distribution stats
            if self.action_dist:
                counts = pd.DataFrame(self.action_dist).apply(lambda x: x.value_counts(), axis=0).fillna(0).to_numpy()
                action_top_k = torch.tensor(counts).topk(min(5, len(counts)), dim=0)
                print(action_top_k.indices)
                print(action_top_k.values / len(self.action_dist))
            self.action_dist.clear()
        self.sim_bridge.wait_action_ready()

        # Get state from shared memory
        values = self.sim_bridge.get_aircraft_state()
        obs = self.get_observation_from_aircraft_state(values)

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
        if self.action_requires_processing:
            action = self.convert_action(action)
        self.action_dist.append(action)

        # Convert action to new format (at least temporarily)
        # Action is length (1 + 72 + 2)
        if action[0] == 0:
            expanded_action = (np.zeros(4 * AIRCRAFT_COUNT, dtype=np.int32))
        else:
            ac_index = action[0] - 1
            expanded_action = np.hstack((np.zeros(ac_index * 4), np.array([action[1], action[2], action[3], 1]), np.zeros((AIRCRAFT_COUNT - 1 - ac_index) * 4))).astype(np.int32)
        # print(expanded_action)

        self.sim_bridge.write_actions(expanded_action)

        # Set the reset request flag before signalling action done
        # The next time the game loop finishes simulating 300 frames, it will stop the update till reset() is called here
        self.steps += 1
        truncated = self.max_steps is not None and self.steps >= self.max_steps
        if truncated and not self.is_eval:
            # print(f"Truncating={truncated}")
            self.sim_bridge.signal_reset_after_step()

        # print(int(time.time() * 1000), "Signalled action done")
        self.sim_bridge.signal_action_done()

        # print(f"{time.time()} Waiting for action ready")

        # Wait till simulator finished simulating 300 frames (action_ready event)
        # print("Waiting for simulation complete")
        self.sim_bridge.wait_action_ready()

        # Read state, reward, terminated, truncated from shared memory
        values = self.sim_bridge.get_total_state()
        # print(values[3 + AIRCRAFT_COUNT * 4:])
        obs = self.get_observation_from_aircraft_state(values[3 + AIRCRAFT_COUNT * 4:])
        reward = values[2]
        terminated = values[1]
        if terminated:
            self.terminated_count += 1

        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        self.sim_bridge.close()
        if self.init_sim:
            print(f"[{self.instance_name}] Ending simulator process")
            self.sim_process.send_signal(signal.CTRL_C_EVENT if platform.system() == "Windows" else signal.SIGINT)


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
            # TODO Set simulator state then step

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


def make_env(
        env_id: int, algo: RLAlgo, ac_type_one_hot_encoder: OneHotEncoder,
        auto_init_sim: bool, reset_print_period: int
):
    backing_env = TC2Env(
        algo, ac_type_one_hot_encoder=ac_type_one_hot_encoder, render_mode="human",
        reset_print_period=reset_print_period, instance_suffix=str(env_id),
        init_sim=auto_init_sim
    )
    return backing_env
