from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt


version = "random-spawn-dir-v2.3c"


# Load Monitor logs
data = load_results(f"./logs/{version}")
timesteps, rewards = ts2xy(data, 'timesteps')

# Plot reward progression
plt.plot(timesteps, rewards)
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("Training Reward Progress")
plt.savefig(f"logs/{version}/training_rewards.png")