from tc2_env import TC2Env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor


version = "v4"


def train():
    backing_env = TC2Env(render_mode="human", reset_print_period=10)
    env = make_vec_env(lambda: Monitor(backing_env), n_envs=1, monitor_dir=f"./logs/{version}")
    print(env.observation_space)
    print(env.action_space)

    model = PPO("MlpPolicy", env, verbose=1, device="cpu", ent_coef=0.01)
    model.learn(total_timesteps=300_000, log_interval=100)
    print("Training done")

    model.save(f"ppo_tc2_{version}")


def run():
    model = PPO.load(f"ppo_tc2_{version}", device="cpu")
    print("Model loaded")

    backing_env = TC2Env(is_eval=True, render_mode="human")
    env = make_vec_env(lambda: backing_env, n_envs=1)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = env.step(action)
        print(reward)
        env.render("human")
        if terminated:
            print("Terminated")
            obs = env.reset()


# train()
run()