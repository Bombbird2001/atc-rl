import ray
from ray.rllib.algorithms.dqn import DQNConfig

def main():
    ray.init(ignore_reinit_error=True)

    config = (
        DQNConfig()
        .environment("CartPole-v1")
        .framework("torch")
        .resources(num_gpus=1)
        .training(
            train_batch_size=32,
            gamma=0.99,
            lr=1e-3,
            replay_buffer_config={
                "type": "PrioritizedEpisodeReplayBuffer",
                "capacity": 5000,
                "alpha": 0.5,
                "beta": 0.5,
            },
            target_network_update_freq=5,
        )
    )

    # Create the trainer
    trainer = config.build_algo()

    # Training loop
    for i in range(20):
        print("Iteration", i + 1)
        trainer.train()

    print(trainer.evaluate())

    # trainer.export_model(export_formats="pytorch", export_dir="./exported_model")

    ray.shutdown()

if __name__ == "__main__":
    main()
