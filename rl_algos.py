import enum
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from typing import Callable, Dict


class RLAlgo:
    def __init__(self, algo_class):
        self.algo_class = algo_class

    def new(self, **kwargs):
        return self.algo_class(**kwargs)

    def load(self, **kwargs):
        return self.algo_class.load(**kwargs)


class PPOWithLogging(PPO):
    def __init__(self, log_stats: Callable[[Dict], None], **kwargs):
        super().__init__(**kwargs)
        self.log_stats = log_stats


    def train(self) -> None:
        super().train()
        self.log_stats(self.logger.name_to_value)


class RLAlgos(enum.Enum):
    PPO = RLAlgo(PPOWithLogging)
    SAC = RLAlgo(SAC)
    PPO_LSTM = RLAlgo(RecurrentPPO)