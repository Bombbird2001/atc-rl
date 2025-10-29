import torch
import torch.nn as nn

from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from torch import Tensor
from typing import Tuple


class MultiAircraftTransformerNetwork(nn.Module):
    def __init__(self,
                 input_dim: int,
                 token_selection_dim: int,
                 action_selection_dim_pi: int,
                 d_model: int = 64,
                 encoder_n_heads: int = 8,
                 encoder_n_layers: int = 3
        ):
        super().__init__()

        # Needed by SB3 to create distributions
        self.input_dim = input_dim
        self.token_selection_dim = token_selection_dim
        self.latent_dim_pi = token_selection_dim + action_selection_dim_pi
        self.latent_dim_vf = 64

        # Project input to a richer representation for attention - outputs (batch_size, n_tokens, d_model)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU()
        )
        # N x Shared self-attention layer with Add & Norm and FFN - outputs (batch_size, n_tokens, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=encoder_n_heads, dim_feedforward=64, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_n_layers)

        # Selecting token to perform action - no pooling to maintain positional equivariance
        self.token_select_net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Action network
        self.action_net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, action_selection_dim_pi)
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.forward_actor(x), self.forward_critic(x)

    def forward_actor(self, x: Tensor) -> Tensor:
        x, attention_mask = self._extract_features(x)
        if x.shape[0] == 1 and attention_mask.all().item() and not self.training:
            return torch.zeros(1, self.latent_dim_pi)
        x = self.input_proj(x)
        x = self.encoder(x, src_key_padding_mask=attention_mask)

        token_select = self.token_select_net(x).squeeze(-1)

        # Pooling to enforce permutation invariance
        action_select = self._mean_pool(x)
        action_select = self.action_net(action_select)
        return torch.concat((token_select, action_select), dim=-1)

    def forward_critic(self, x: Tensor) -> Tensor:
        x, attention_mask = self._extract_features(x)
        if x.shape[0] == 1 and attention_mask.all().item() and not self.training:
            return torch.Tensor([0])
        x = self.input_proj(x)
        x = self.encoder(x, src_key_padding_mask=attention_mask)

        # Pooling to enforce permutation invariance
        x = self._mean_pool(x)
        return self.value_net(x)

    def _extract_features(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = x.reshape(-1, self.token_selection_dim, self.input_dim + 1)
        attention_mask = 1 - x[:,:,-1]  # In TransformerEncoderLayer, 1 is used for non-existent tokens
        x = x[:,:,:-1]
        return x, attention_mask

    @staticmethod
    def _mean_pool(x: Tensor) -> Tensor:
        # Mean pooling along token dimensions - outputs (batch_size, d_model)
        return x.mean(dim=1)


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