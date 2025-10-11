import os
import time
import wandb

from callbacks import PPOStatsCallback
from playsound3 import playsound
from rl_algos import RLAlgos
from stable_baselines3.common.env_util import make_vec_env
from tc2_env import make_env


ALGO = RLAlgos.PPO
algo = ALGO.value
algo_name = ALGO.name
TRAIN = True
ENV_COUNT = 4
if ALGO == RLAlgos.SAC:
    LEARNING_RATE = 5e-4
    MIN_LR = 1e-5
    TIMESTEPS = 400_000
    POLICY_NAME = "MlpPolicy"
    model_kwargs = {
        "ent_coef": "auto",
        "batch_size": 256,
        "gamma": 1
    }
    STATS_LOG_INTERVAL = 100
elif ALGO == RLAlgos.PPO:
    LEARNING_RATE = 2e-4
    MIN_LR = 5e-6
    TIMESTEPS = 50_000
    POLICY_NAME = "MlpPolicy"
    model_kwargs = {
        "ent_coef": 0.03,
        "n_epochs": 5,
        "batch_size": 64,
        "gamma": 1
    }
    STATS_LOG_INTERVAL = 20
elif ALGO == RLAlgos.PPO_LSTM:
    LEARNING_RATE = 2e-4
    MIN_LR = 1e-5
    TIMESTEPS = 1_000_000
    POLICY_NAME = "MlpLstmPolicy"
    model_kwargs = {
        "ent_coef": 0.04,
        "n_epochs": 10,
        "batch_size": 128,
        "gamma": 1
    }
    STATS_LOG_INTERVAL = 50
else:
    raise NotImplementedError(f"Unknown policy {ALGO.name}")
DEVICE = "cpu"
AUTO_INIT_SIM = True
start_from_version = None
# version = f"random-spawn-dir-v1.0-no-clearance-penalty-lr-{LEARNING_RATE}-ent-coef-{ENTROPY_COEF}-steps-{TIMESTEPS}"
version = "test"


if not AUTO_INIT_SIM:
    ENV_COUNT = 1


def linear_schedule(initial_value: float, min_lr: float):
    def func(progress_remaining: float) -> float:
        return max(progress_remaining * initial_value, min_lr)
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

    wandb_run = wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        config={
            "env_count": ENV_COUNT,
            "learning_rate": LEARNING_RATE,
            "min_lr": MIN_LR,
            "timesteps": TIMESTEPS,
            "policy_name": POLICY_NAME,
            **model_kwargs
        }
    )

    if start_from_version is not None:
        model = algo.load(
            path=f"{algo_name}/{algo_name}_tc2_{start_from_version}", env=tc2_env, verbose=1, device=DEVICE,
            learning_rate=linear_schedule(LEARNING_RATE, MIN_LR), log_stats=wandb_run.log, **model_kwargs
        )
    else:
        model = algo.new(
            policy=POLICY_NAME, env=tc2_env, verbose=1, device=DEVICE,
            learning_rate=linear_schedule(LEARNING_RATE, MIN_LR), log_stats=wandb_run.log, **model_kwargs
        )
    start_time = time.time()
    model.learn(total_timesteps=TIMESTEPS, log_interval=STATS_LOG_INTERVAL, callback=PPOStatsCallback(log_stats=wandb_run.log, log_interval=2))
    end_time = time.time()
    print(f"Training done in {((end_time - start_time) // 60):.0f}m {((end_time - start_time) % 60):.2f}s")

    model_file = f"{algo_name}/{algo_name}_tc2_{version}.zip"
    model.save(model_file)
    print("Output to", f"{algo_name}/logs/{version}")

    artifact = wandb.Artifact(name="Model", type="model")
    artifact.add_file(local_path=model_file)
    artifact.save()

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