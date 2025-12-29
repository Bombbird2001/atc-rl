import torch.nn.functional as F
from torch.nn import Module, Linear, LayerNorm, LeakyReLU, Sequential
from torch_geometric.nn import GINEConv


HDG_BINS = 360 // 5


class WSSSAPP02GINE(Module):
    def __init__(self, node_feature_count, edge_feature_count):
        super().__init__()

        nn1 = Sequential(
            Linear(node_feature_count, 32),
            LayerNorm(32),
            LeakyReLU(0.1),
            Linear(32, 16),
        )

        self.gine1 = GINEConv(nn1, edge_dim=edge_feature_count, train_eps=True)
        self.ln = LayerNorm(16)
        # 72 bins for heading, 1 output for altitude, 1 output for speed
        self.linear = Linear(16, HDG_BINS + 1 + 1)

    def forward(self, x, edge_index, edge_attr):
        # print(x.dtype, edge_index.dtype, edge_attr.dtype)
        h = self.gine1(x, edge_index, edge_attr)
        h = self.ln(h)
        h = F.leaky_relu(h, 0.1)
        h = self.linear(h)

        return h

    @property
    def name(self):
        return f"gine1_linear1"