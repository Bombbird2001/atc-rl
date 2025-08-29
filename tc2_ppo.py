from tc2_env import TC2Env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor


start_from_version = "v3.3"
version = "v3.4"


def train():
    backing_env = TC2Env(render_mode="human", reset_print_period=10)
    tc2_env = make_vec_env(lambda: Monitor(backing_env), n_envs=1, monitor_dir=f"./logs/{version}")
    print(tc2_env.observation_space)
    print(tc2_env.action_space)

    if start_from_version is not None:
        model = PPO.load(f"ppo_tc2_{start_from_version}", env=tc2_env, verbose=1, device="cpu", ent_coef=0.01)
    else:
        model = PPO("MlpPolicy", tc2_env, verbose=1, device="cpu", ent_coef=0.01)
    model.learn(total_timesteps=200_000, log_interval=100)
    print("Training done")

    model.save(f"ppo_tc2_{version}.zip")


def run():
    model = PPO.load(f"ppo_tc2_{version}", device="cpu")
    print("Model loaded")

    backing_env = TC2Env(is_eval=True, render_mode="human")
    tc2_eval_env = make_vec_env(lambda: backing_env, n_envs=1)
    obs = tc2_eval_env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = tc2_eval_env.step(action)
        tc2_eval_env.render("human")
        if terminated:
            print("Terminated")
            obs = tc2_eval_env.reset()


# train()
run()