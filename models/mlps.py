import torch.nn.functional as F
from common.constants import HDG_BINS, ALT_BINS, SPD_BINS
from torch.nn import Module, Linear


class WSSSAPP02MLP(Module):
    def __init__(self, node_feature_count, edge_feature_count):
        super().__init__()

        self.linear1 = Linear(node_feature_count, 64)
        self.linear2 = Linear(64, 3 + HDG_BINS + ALT_BINS + SPD_BINS)

    def forward(self, x, edge_index, edge_attr):
        h = self.linear1(x)
        h = F.tanh(h)
        latent = h
        h = self.linear2(h)

        return h, latent


class WSSSAPP02MLPValueNet(Module):
    def __init__(self, node_feature_count, edge_feature_count):
        super().__init__()

        self.linear1 = Linear(node_feature_count, 64)
        self.linear2 = Linear(64, 1)

    def forward(self, x, edge_index, edge_attr):
        h = self.linear1(x)
        h = F.tanh(h)
        h = self.linear2(h)

        return h