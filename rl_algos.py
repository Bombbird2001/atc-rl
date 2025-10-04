import enum
from stable_baselines3 import PPO, SAC


class RLAlgo:
    def __init__(self, algo_class):
        self.algo_class = algo_class

    def new(self, **kwargs):
        return self.algo_class(**kwargs)

    def load(self, **kwargs):
        self.algo_class.load(**kwargs)


class RLAlgos(enum.Enum):
    PPO = RLAlgo(PPO)
    SAC = RLAlgo(SAC)