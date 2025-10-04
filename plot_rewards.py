import matplotlib.pyplot as plt

from rl_algos import RLAlgos
from stable_baselines3.common.results_plotter import load_results, ts2xy


algo = RLAlgos.SAC.name
version = "random-spawn-dir-v2.0-large-change-penalty"


# Load Monitor logs
data = load_results(f"./{algo}/logs/{version}")
timesteps, rewards = ts2xy(data, 'timesteps')

# Plot reward progression
plt.plot(timesteps, rewards)
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("Training Reward Progress")
plt.savefig(f"{algo}/logs/{version}/training_rewards.png")