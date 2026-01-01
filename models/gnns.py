import torch
import torch.nn.functional as F
from constants import HDG_BINS
from torch.nn import Module, Linear, LayerNorm, GELU, Sequential
from torch_geometric.nn import GINEConv


class WSSSAPP02GINE(Module):
    def __init__(self, node_feature_count, edge_feature_count):
        super().__init__()

        nn1 = Sequential(
            Linear(node_feature_count, 64),
            LayerNorm(64),
            GELU(),
            Linear(64, 32),
        )
        # nn2 = Sequential(
        #     Linear(32, 32),
        #     LayerNorm(32),
        #     GELU(),
        #     Linear(32, 16),
        # )

        self.gine1 = GINEConv(nn1, edge_dim=edge_feature_count, train_eps=True)
        # self.gine2 = GINEConv(nn2, edge_dim=edge_feature_count, train_eps=True)
        self.ln1 = LayerNorm(32)
        # self.ln2 = LayerNorm(16)
        # 3 outputs for probability of changing each clearance, 72 bins for heading, 1 output for altitude, 1 output for speed
        self.linear = Linear(32, 3 + HDG_BINS + 1 + 1)

    def forward(self, x, edge_index, edge_attr):
        h = self.gine1(x, edge_index, edge_attr)
        h = self.ln1(h)
        h = F.gelu(h)
        latent = h
        # h = self.gine2(h, edge_index, edge_attr)
        # h = self.ln2(h)
        # h = F.gelu(h)
        h = self.linear(h)

        # Also return latent representation to pass to separate value/critic net
        return h, latent

    @property
    def name(self):
        return f"gine1_linear1"


class WSSSAPP02ValueNet(Module):
    def __init__(self):
        super().__init__()

        self.linear = Linear(32, 1)

    def forward(self, x: torch.Tensor):
        # Global average pooling
        h = x.mean(dim=0)
        h = self.linear(h)

        return h

    @property
    def name(self):
        return f"linear1"