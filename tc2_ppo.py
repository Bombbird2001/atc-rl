import joblib
import os
import time
import torch
import wandb

from callbacks import PPOStatsCallback
from common.data_preprocessing import TransformerProcessor, GNNProcessor
from constants import AIRCRAFT_COUNT
from datetime import datetime
from models.encoders import WSSSAPP02Encoder
from models.gnns import WSSSAPP02GINE
from playsound3 import playsound
from policies import MultiAircraftTransformerPolicy
from rl_algos import RLAlgos
from stable_baselines3.common.env_util import make_vec_env
from tc2_env import make_env


ALGO = RLAlgos.PPO
algo = ALGO.value
algo_name = ALGO.name

if ALGO == RLAlgos.SAC:
    LEARNING_RATE = 5e-4
    MIN_LR = 1e-5
    TIMESTEPS = 400_000
    POLICY = "MlpPolicy"
    model_kwargs = {
        "ent_coef": "auto",
        "batch_size": 256,
        "gamma": 0.99
    }
    STATS_LOG_INTERVAL = 100
elif ALGO == RLAlgos.PPO:
    LEARNING_RATE = 7.5e-5
    MIN_LR = LEARNING_RATE * 0.2
    TIMESTEPS = 200_000
    POLICY = MultiAircraftTransformerPolicy
    model_kwargs = {
        "ent_coef": 0.01,
        "n_epochs": 5,
        "n_steps": 256,
        "batch_size": 1024,
        "gamma": 0.99,
        "policy_kwargs": {
            "token_dim": 11,
            "max_tokens": AIRCRAFT_COUNT,
        },
    }
    STATS_LOG_INTERVAL = 3
elif ALGO == RLAlgos.PPO_LSTM:
    LEARNING_RATE = 2e-4
    MIN_LR = 1e-5
    TIMESTEPS = 1_000_000
    POLICY = "MlpLstmPolicy"
    model_kwargs = {
        "ent_coef": 0.04,
        "n_epochs": 10,
        "batch_size": 128,
        "gamma": 0.99
    }
    STATS_LOG_INTERVAL = 50
else:
    raise NotImplementedError(f"Unknown policy {ALGO.name}")


TRAIN = False
ENV_COUNT = 128
DEVICE = "cpu"
AUTO_INIT_SIM = True
start_from_version = None
version = f"multi-aircraft-transformer-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-lr-{LEARNING_RATE}-ent-coef-{model_kwargs['ent_coef']}-steps-{TIMESTEPS}"
additional_description = f"""Random spawn location
No clearance penalty
No conflict enforcement
Max offset angle 80 degrees for LOC capture
Altitude below G/S for LOC capture
No max IAS for LOC capture"""
# version = "multi-aircraft-test"
eval_version = "multi-aircraft-transformer-2025-11-01_21-50-28-lr-0.0001-ent-coef-0.03-steps-600000"


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
        name=f"{algo_name}-{version}",
        config={
            "description": additional_description,
            "started_from_version": start_from_version,
            "env_count": ENV_COUNT,
            "learning_rate": LEARNING_RATE,
            "min_lr": MIN_LR,
            "timesteps": TIMESTEPS,
            "policy_name": POLICY if isinstance(POLICY, str) else POLICY.__name__,
            **model_kwargs
        }
    )

    with open(f"./{algo_name}/logs/{version}/desc.txt", "w") as f:
        f.write(additional_description)

    if start_from_version is not None:
        model = algo.load(
            path=f"{algo_name}/{algo_name}_tc2_{start_from_version}", env=tc2_env, verbose=1, device=DEVICE,
            learning_rate=linear_schedule(LEARNING_RATE, MIN_LR), log_stats=wandb_run.log, **model_kwargs
        )
    else:
        model = algo.new(
            policy=POLICY, env=tc2_env, verbose=1, device=DEVICE,
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
    NODE_FEATURE_DIM = 32

    # GINE only
    EDGE_FEATURE_DIM = 2

    # Transformer only
    # D_MODEL = 32
    # N_HEAD = 1
    # N_LAYERS = 3

    # processor = TransformerProcessor()
    processor = GNNProcessor()

    # model = WSSSAPP02Encoder(NODE_FEATURE_DIM, D_MODEL, N_HEAD, N_LAYERS, AIRCRAFT_COUNT)
    model = WSSSAPP02GINE(NODE_FEATURE_DIM, EDGE_FEATURE_DIM)
    model.load_state_dict(torch.load("C:\\IdeaProjects\\atc-rl-adsbexchange\\trained_models\\feat32_gine1_linear1_Adam_lr-0.01_batch_32_epochs-25_2025-12-30_112300\\15.pt"))
    model.eval()
    # print(model)
    print("Model loaded")

    tc2_eval_env = make_vec_env(make_env, n_envs=1,
                                env_kwargs={
                                    "algo": ALGO,
                                    "ac_type_one_hot_encoder": joblib.load("common/ac_type_one_hot_encoder.joblib"),
                                    "auto_init_sim": False,
                                    "reset_print_period": 1,
                                })
    obs = tc2_eval_env.reset()
    cumulative_reward = 0
    while True:
        with torch.no_grad():
            # x, attention_mask = processor.preprocess_data(obs)
            # print(x[0], attention_mask)
            # action = model(x, attention_mask)
            # action = processor.postprocess_data(action, attention_mask)

            x = processor.preprocess_data(torch.Tensor(obs))
            action = model(x.x, x.edge_index, x.edge_attr)[0]
            action = processor.postprocess_data_multi_aircraft(action)
            # print(action)
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