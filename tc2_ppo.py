from tc2_env import TC2Env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def train():
    backing_env = TC2Env(render_mode="human")
    env = make_vec_env(lambda: backing_env, n_envs=1)
    print(env.observation_space)
    print(env.action_space)

    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=100_000, log_interval=100)
    print("Training done")

    model.save("ppo_tc2_v2")


def run():
    model = PPO.load("ppo_tc2_v2", device="cpu")
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