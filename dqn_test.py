import gymnasium as gym

from stable_baselines3 import PPO


def train():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    print(env.observation_space)
    print(env.action_space)

    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=50_000, log_interval=10)
    # model.save("ppo_cartpole")

    # del model

    # model = PPO.load("ppo_cartpole")

    env = model.get_env()
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = env.step(action)
        env.render("human")
        if terminated:
            obs = env.reset()


train()