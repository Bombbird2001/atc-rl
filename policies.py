import torch
import torch as th
import torch.nn as nn

from common.data_preprocessing import GNNProcessor
from gymnasium import spaces
from models.gnns import WSSSAPP02GINE, WSSSAPP02ValueNet
from models.old_models import MultiAircraftTransformerNetwork
from stable_baselines3.common.distributions import DiagGaussianDistribution, CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from typing import Optional


class MultiAircraftTransformerPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        token_dim: int,
        max_tokens: int,
        *args,
        **kwargs,
    ):
        kwargs["ortho_init"] = False

        self.token_dim = token_dim
        self.max_tokens = max_tokens

        if (self.token_dim + 1) * self.max_tokens != observation_space.shape[0]:
            raise ValueError(f"Observation space shape {observation_space.shape[0]} does not match (token_dim * (max_tokens + 1)) = {self.token_dim} * {self.max_tokens + 1}")

        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MultiAircraftTransformerNetwork(
            input_dim=self.token_dim,
            token_selection_dim=self.max_tokens,
            action_selection_dim_pi=self.action_space.nvec.sum() - self.max_tokens
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        self.action_net = nn.Identity()
        self.value_net = nn.Identity()

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


class MultiAircraftGNNPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            node_feature_dim: int,
            edge_feature_dim: int,
            lr_schedule: Schedule,
            load_model_path: Optional[str] = None,
            *args,
            **kwargs,
    ):
        kwargs["ortho_init"] = False

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.load_model_path = load_model_path

        self._init_distributions(action_space)

        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_gnn_extractor(self) -> None:
        self.gnn_model = WSSSAPP02GINE(self.node_feature_dim, self.edge_feature_dim)
        if self.load_model_path is not None:
            self.gnn_model.load_state_dict(torch.load(self.load_model_path))

    def _build(self, lr_schedule: Schedule) -> None:
        self.feature_processor = GNNProcessor()

        self._build_gnn_extractor()

        self.action_net = nn.Identity()
        self.value_net = WSSSAPP02ValueNet()

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(list(self.gnn_model.parameters()) + list(self.value_net.parameters()), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _init_distributions(self, action_space: spaces.Space):
        self.aircraft_dist = CategoricalDistribution(action_space.nvec[0])
        self.hdg_dist = CategoricalDistribution(action_space.nvec[1])
        self.alt_spd_dist = DiagGaussianDistribution(2)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        x = self.feature_processor.preprocess_data(obs)
        actions_raw, latent_rep = self.gnn_model(x.x, x.edge_index, x.edge_attr)
        aircraft_logits, action_logits = self.feature_processor.postprocess_data(actions_raw)

        ac_dist = self.aircraft_dist.proba_distribution(aircraft_logits)
        # print(ac_dist.distribution.probs)
        ac_index = ac_dist.get_actions(deterministic=deterministic)
        actions = torch.Tensor([ac_index, 0, 0, 0])
        log_prob = ac_dist.log_prob(ac_index)

        if actions[0] >= 1:
            hdg_dist = self.aircraft_dist.proba_distribution(action_logits[ac_index - 1, :72])
            # print(hdg_dist.distribution.probs)
            hdg_index = hdg_dist.get_actions(deterministic=deterministic)
            actions[1] = hdg_index
            log_prob += hdg_dist.log_prob(hdg_index)

            # TODO Add the log_probs for continuous variables (needs Gaussian distribution)
            # Integrate over bin range, then divide over integration over valid range [-1, 1]

        values = self.value_net(latent_rep)

        return actions, values, log_prob