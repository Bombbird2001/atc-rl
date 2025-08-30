import time

from tc2_env import TC2Env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


ENV_COUNT = 1
ENTROPY_COEF = 0.01
DEVICE = "cpu"
start_from_version = "v3.4"
version = "v3.5"


def make_env(env_id):
    backing_env = TC2Env(render_mode="human", reset_print_period=10, instance_suffix=env_id)
    return backing_env


def train():
    tc2_env = make_vec_env(make_env, n_envs=ENV_COUNT, monitor_dir=f"./logs/{version}")
    print("State space:", tc2_env.observation_space)
    print("Action space", tc2_env.action_space)
    print("Using device:", DEVICE)

    if start_from_version is not None:
        model = PPO.load(f"ppo_tc2_{start_from_version}", env=tc2_env, verbose=1, device=DEVICE, ent_coef=ENTROPY_COEF)
    else:
        model = PPO("MlpPolicy", tc2_env, verbose=1, device=DEVICE, ent_coef=ENTROPY_COEF)
    start_time = time.time()
    model.learn(total_timesteps=200_000, log_interval=100)
    end_time = time.time()
    print(f"Training done in {((end_time - start_time) // 60):.0f}m {((end_time - start_time) % 60):.2f}s")

    model.save(f"ppo_tc2_{version}.zip")


def run():
    model = PPO.load(f"ppo_tc2_{version}", device="cpu")
    print("Model loaded")

    backing_env = TC2Env(is_eval=True, render_mode="human", instance_suffix="0")
    tc2_eval_env = make_vec_env(lambda: backing_env, n_envs=1)
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