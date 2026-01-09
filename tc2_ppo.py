import argparse
import joblib
import numpy as np
import os
import time
import torch
import wandb
from common.constants import AIRCRAFT_COUNT, TEST_DATA
from datetime import datetime
from gymnasium import spaces
from models.models import ActionNets, ValueNets
from playsound3 import playsound
from policies import MultiAircraftGraphPolicy
from rl_algos import RLAlgos
from stable_baselines3.common.env_util import make_vec_env
from envs.tc2_gym_env import make_env
from utils.callbacks import PPOStatsCallback


TRAIN = True
ENV_COUNT = 8
DEVICE = "cpu"
AUTO_INIT_SIM = True
start_from_version = None
additional_description = f"""STAR spawn location
No clearance penalty
No conflict enforcement
Max offset angle 80 degrees for LOC capture
Altitude below G/S for LOC capture
No max IAS for LOC capture"""


if not AUTO_INIT_SIM:
    ENV_COUNT = 1

def linear_schedule(initial_value: float, min_lr: float):
    def func(progress_remaining: float) -> float:
        return max(progress_remaining * initial_value, min_lr)
    return func


def train(
        algo_name: str, action_model: str, value_model: str, learning_rate: float, timesteps: int, ent_coef: float,
        n_epochs: int, n_steps: int, batch_size: int, gamma: float
):
    algo = RLAlgos[algo_name]

    min_lr = learning_rate * 0.2

    if algo == RLAlgos.SAC:
        POLICY = "MlpPolicy"
        model_kwargs = {
            "ent_coef": "auto",
            "batch_size": batch_size,
            "gamma": gamma
        }
        STATS_LOG_INTERVAL = 100
    elif algo == RLAlgos.PPO:
        POLICY = MultiAircraftGraphPolicy
        model_kwargs = {
            "ent_coef": ent_coef,
            "n_epochs": n_epochs,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "gamma": gamma,
            "policy_kwargs": {
                # "token_dim": 11,
                # "max_tokens": AIRCRAFT_COUNT,
                "action_model_class": ActionNets[action_model].value,
                "value_net_class": ValueNets[value_model].value,
                "node_feature_dim": 18,
                "edge_feature_dim": 2,
                "freeze_action_net": False,
                # "load_model_path": "/Users/bombbird2001/Desktop/atc-rl-adsbexchange/trained_models/feat18_gine2_linear1_Adam_lr-0.005_batch_32_epochs-50_2026-01-07_072725/29.pt",
            },
        }
        STATS_LOG_INTERVAL = 3
    elif algo == RLAlgos.PPO_LSTM:
        POLICY = "MlpLstmPolicy"
        model_kwargs = {
            "ent_coef": ent_coef,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "gamma": gamma
        }
        STATS_LOG_INTERVAL = 50
    else:
        raise NotImplementedError(f"Unknown policy {algo_name}")

    version = f"multi-aircraft-gnn-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-lr-{learning_rate}-batch-{model_kwargs['batch_size']}-ent-coef-{model_kwargs['ent_coef']}-steps-{timesteps}"

    tc2_env = make_vec_env(make_env, n_envs=ENV_COUNT,
                           env_kwargs={
                               "algo": algo,
                               "ac_type_one_hot_encoder": joblib.load("common/recat_one_hot_encoder.joblib"),
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
            "learning_rate": learning_rate,
            "min_lr": min_lr,
            "timesteps": timesteps,
            "policy_name": POLICY if isinstance(POLICY, str) else POLICY.__name__,
            **model_kwargs
        }
    )

    with open(f"./{algo_name}/logs/{version}/desc.txt", "w") as f:
        f.write(additional_description)

    if start_from_version is not None:
        model = algo.value.load(
            path=f"{algo_name}/{algo_name}_tc2_{start_from_version}", env=tc2_env, verbose=1, device=DEVICE,
            learning_rate=linear_schedule(learning_rate, min_lr), log_stats=wandb_run.log, **model_kwargs
        )
    else:
        model = algo.value.new(
            policy=POLICY, env=tc2_env, verbose=1, device=DEVICE,
            learning_rate=linear_schedule(learning_rate, min_lr), log_stats=wandb_run.log, **model_kwargs
        )
    start_time = time.time()
    model.learn(total_timesteps=timesteps, log_interval=STATS_LOG_INTERVAL, callback=PPOStatsCallback(log_stats=wandb_run.log, log_interval=2))
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
    NODE_FEATURE_DIM = 18

    # GINE only
    EDGE_FEATURE_DIM = 2

    # Transformer only
    # D_MODEL = 32
    # N_HEAD = 1
    # N_LAYERS = 3

    policy = MultiAircraftGraphPolicy(
        spaces.Box(
            low=np.repeat(-1.0, 33 * AIRCRAFT_COUNT),
            high=np.repeat(1.0, 33 * AIRCRAFT_COUNT),
            dtype=np.float32
        ), spaces.MultiDiscrete([1 + AIRCRAFT_COUNT, 72, 14, 10]),
        lambda x: 1, ActionNets.WSSSAPP02GINE.value, ValueNets.WSSSAPP02GINEValueNet.value, NODE_FEATURE_DIM, EDGE_FEATURE_DIM,
        load_model_path="/Users/bombbird2001/Desktop/atc-rl-adsbexchange/trained_models/feat18_gine2_linear1_Adam_lr-0.005_batch_32_epochs-50_2026-01-04_044719/21.pt"
    )
    policy.eval()
    print("Model loaded")

    tc2_eval_env = make_vec_env(make_env, n_envs=1,
                                env_kwargs={
                                    "algo": RLAlgos.PPO,
                                    "ac_type_one_hot_encoder": joblib.load("common/recat_one_hot_encoder.joblib"),
                                    "auto_init_sim": False,
                                    "reset_print_period": 1,
                                })
    obs = tc2_eval_env.reset()
    cumulative_reward = 0
    while True:
        with torch.no_grad():
            action = policy.forward(torch.Tensor(obs), deterministic=True)[0].numpy().astype(np.int32)
            obs, reward, terminated, info = tc2_eval_env.step(action)
            cumulative_reward += reward
            if terminated:
                print("Total reward:", cumulative_reward)
                cumulative_reward = 0


def quick_test():
    policy = MultiAircraftGraphPolicy(
        spaces.Box(
            low=np.repeat(-1.0, 33 * AIRCRAFT_COUNT),
            high=np.repeat(1.0, 33 * AIRCRAFT_COUNT),
            dtype=np.float32
        ), spaces.MultiDiscrete([1 + AIRCRAFT_COUNT, 72, 14, 10]),
        lambda x: 1, ActionNets.WSSSAPP02GINE.value, ValueNets.WSSSAPP02GINEValueNet.value, 18, 2,
        load_model_path="/Users/bombbird2001/Desktop/atc-rl-adsbexchange/trained_models/feat18_gine2_linear1_Adam_lr-0.005_batch_32_epochs-50_2026-01-04_044719/21.pt"
    )

    action_1, value_1, log_prob_1 = policy(TEST_DATA, deterministic=False)
    action_2, value_2, log_prob_2 = policy(TEST_DATA, deterministic=False)
    print(action_1, value_1, log_prob_1)
    print(action_2, value_2, log_prob_2)
    actions = torch.vstack((action_1, action_2))
    values, log_probs, entropies = policy.evaluate_actions(torch.vstack([TEST_DATA, TEST_DATA]), actions)
    print(actions, values, log_probs, entropies)
    stuff = policy(torch.vstack([TEST_DATA, TEST_DATA]), deterministic=False)
    print(stuff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=['PPO'], type=str)
    parser.add_argument('--action_model', type=str)
    parser.add_argument('--value_model', type=str)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--timesteps', type=int)
    parser.add_argument('--ent_coef', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--n_steps', type=int)
    parser.add_argument('--gamma', type=float)
    args = parser.parse_args()

    print(args)

    if TRAIN:
        train(
            args.algo, args.action_model, args.value_model, args.learning_rate, args.timesteps, args.ent_coef,
            args.n_epochs, args.n_steps, args.batch_size, args.gamma
        )
    else:
        # quick_test()
        run()
