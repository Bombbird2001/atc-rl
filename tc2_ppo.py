import time

from playsound3 import playsound
from rl_algos import RLAlgos
from stable_baselines3.common.env_util import make_vec_env
from tc2_env import make_env


ALGO = RLAlgos.SAC
algo = ALGO.value
algo_name = ALGO.name
TRAIN = True
ENV_COUNT = 4
if ALGO == RLAlgos.SAC:
    ENTROPY_COEF = "auto"
    LEARNING_RATE = 5e-4
    TIMESTEPS = 400_000
    STATS_LOG_INTERVAL = 100
elif ALGO == RLAlgos.PPO:
    ENTROPY_COEF = 0.03
    LEARNING_RATE = 2e-4
    TIMESTEPS = 800_000
    STATS_LOG_INTERVAL = 20
GAMMA = 1
DEVICE = "cpu"
AUTO_INIT_SIM = True
start_from_version = None
version = "random-spawn-dir-v1.0-no-obstacle"


if not AUTO_INIT_SIM:
    ENV_COUNT = 1


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def train():
    tc2_env = make_vec_env(make_env, n_envs=ENV_COUNT,
                           env_kwargs={
                               "algo": ALGO,
                               "auto_init_sim": TRAIN and AUTO_INIT_SIM,
                               "reset_print_period": 50,
                           }, monitor_dir=f"./{algo_name}/logs/{version}"
                           )
    print("State space:", tc2_env.observation_space)
    print("Action space", tc2_env.action_space)

    if start_from_version is not None:
        model = algo.load(
            path=f"{algo_name}/{algo_name}_tc2_{start_from_version}", env=tc2_env, verbose=1, device=DEVICE,
            ent_coef=ENTROPY_COEF, learning_rate=linear_schedule(LEARNING_RATE), gamma=GAMMA
        )
    else:
        model = algo.new(
            policy="MlpPolicy", env=tc2_env, verbose=1, device=DEVICE, ent_coef=ENTROPY_COEF,
            learning_rate=linear_schedule(LEARNING_RATE), gamma=GAMMA
        )
    start_time = time.time()
    model.learn(total_timesteps=TIMESTEPS, log_interval=STATS_LOG_INTERVAL)
    end_time = time.time()
    print(f"Training done in {((end_time - start_time) // 60):.0f}m {((end_time - start_time) % 60):.2f}s")

    model.save(f"{algo_name}/{algo_name}_tc2_{version}.zip")

    tc2_env.close()

    playsound("sounds/alert.mp3")


def run():
    model = algo.load(path=f"{algo_name}/{algo_name}_tc2_{version}", device="cpu")
    print("Model loaded")

    tc2_eval_env = make_vec_env(make_env, n_envs=1,
                                env_kwargs={
                                    "algo": ALGO,
                                    "auto_init_sim": False,
                                    "reset_print_period": 1,
                                })
    obs = tc2_eval_env.reset()
    cumulative_reward = 0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = tc2_eval_env.step(action)
        cumulative_reward += reward
        if terminated:
            print("Total reward:", cumulative_reward)
            cumulative_reward = 0


if __name__ == "__main__":
    if TRAIN:
        train()
    else:
        run()