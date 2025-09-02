import signal
import time

from tc2_env import TC2Env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from typing import List


ENV_COUNT = 1
ENTROPY_COEF = 0.01
TIMESTEPS = 200_000
DEVICE = "cpu"
AUTO_INIT_SIM = True
start_from_version = "v4.1"
version = "v4.2"


if not AUTO_INIT_SIM:
    ENV_COUNT = 1


def make_env(env_id: int, processes: List, auto_init_sim: bool):
    backing_env = TC2Env(render_mode="human", reset_print_period=20, instance_suffix=env_id, init_sim=auto_init_sim)
    if AUTO_INIT_SIM:
        processes.append(backing_env.sim_process)
    return backing_env


def train():
    processes_to_kill = []
    tc2_env = make_vec_env(make_env, n_envs=ENV_COUNT,
                           env_kwargs={"processes": processes_to_kill, "auto_init_sim": AUTO_INIT_SIM}, monitor_dir=f"./logs/{version}"
                           )
    print("State space:", tc2_env.observation_space)
    print("Action space", tc2_env.action_space)

    if start_from_version is not None:
        model = PPO.load(f"ppo_tc2_{start_from_version}", env=tc2_env, verbose=1, device=DEVICE, ent_coef=ENTROPY_COEF)
    else:
        model = PPO("MlpPolicy", tc2_env, verbose=1, device=DEVICE, ent_coef=ENTROPY_COEF, learning_rate=5e-5)
    start_time = time.time()
    model.learn(total_timesteps=TIMESTEPS, log_interval=20)
    end_time = time.time()
    print(f"Training done in {((end_time - start_time) // 60):.0f}m {((end_time - start_time) % 60):.2f}s")

    model.save(f"ppo_tc2_{version}.zip")

    if AUTO_INIT_SIM:
        print("Ending simulator process(es)")
        for process in processes_to_kill:
            process.send_signal(signal.CTRL_C_EVENT)


def run():
    model = PPO.load(f"ppo_tc2_{version}", device="cpu")
    print("Model loaded")

    tc2_eval_env = make_vec_env(make_env, n_envs=1, env_kwargs={"processes": [], "auto_init_sim": False})
    obs = tc2_eval_env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = tc2_eval_env.step(action)
        if terminated:
            print("Terminated")
            obs = tc2_eval_env.reset()


if __name__ == "__main__":
    train()
    # run()