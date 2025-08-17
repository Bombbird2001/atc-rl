import gymnasium as gym
from custom_env import CounterEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def train():
    # env = gym.make("CartPole-v1", render_mode="rgb_array")
    backing_env = CounterEnv(target=5, render_mode="human")
    env = make_vec_env(lambda: backing_env, n_envs=1)
    print(env.observation_space)
    print(env.action_space)

    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=5_000, log_interval=100)
    print("Training done")

    model.save("ppo_counter")
    del model
    model = PPO.load("ppo_counter", device="cpu")
    print("Model loaded")

    # env = model.get_env()
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = env.step(action)
        env.render("human")
        if terminated:
            obs = env.reset()


train()