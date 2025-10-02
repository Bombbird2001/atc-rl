import platform
import signal
import time

from tc2_env import make_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


TRAIN = True
ENV_COUNT = 1
ENTROPY_COEF = 0.02
LEARNING_RATE = 2e-4
TIMESTEPS = 1_000_000
DEVICE = "cpu"
AUTO_INIT_SIM = True
start_from_version = None
version = "random-spawn-dir-v1.0"


if not AUTO_INIT_SIM:
    ENV_COUNT = 1


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def train():
    processes_to_kill = []
    tc2_env = make_vec_env(make_env, n_envs=ENV_COUNT,
                           env_kwargs={"processes": processes_to_kill, "auto_init_sim": TRAIN and AUTO_INIT_SIM}, monitor_dir=f"./logs/{version}"
                           )
    print("State space:", tc2_env.observation_space)
    print("Action space", tc2_env.action_space)

    if start_from_version is not None:
        model = PPO.load(f"ppo_tc2_{start_from_version}", env=tc2_env, verbose=1, device=DEVICE, ent_coef=ENTROPY_COEF, learning_rate=linear_schedule(LEARNING_RATE))
    else:
        model = PPO("MlpPolicy", tc2_env, verbose=1, device=DEVICE, ent_coef=ENTROPY_COEF, learning_rate=linear_schedule(LEARNING_RATE))
    start_time = time.time()
    model.learn(total_timesteps=TIMESTEPS, log_interval=20)
    end_time = time.time()
    print(f"Training done in {((end_time - start_time) // 60):.0f}m {((end_time - start_time) % 60):.2f}s")

    model.save(f"ppo_tc2_{version}.zip")

    if AUTO_INIT_SIM:
        print("Ending simulator process(es)")
        for process in processes_to_kill:
            process.send_signal(signal.CTRL_C_EVENT if platform.system() == "Windows" else signal.SIGINT)


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
    if TRAIN:
        train()
    else:
        run()