from constants import HDG_BINS
from torch.nn import Module, Sequential, Linear, GELU, TransformerEncoder, TransformerEncoderLayer, BatchNorm1d


class WSSSAPP02Encoder(Module):
    def __init__(self, node_feature_count, d_model, n_head, num_layers, max_seq_length):
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.num_layers = num_layers
        self.max_seq_len = max_seq_length

        self.input_proj = Sequential(
            Linear(node_feature_count, d_model),
            GELU()
        )
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=32, activation='gelu', batch_first=True)
        self.bn1 = BatchNorm1d(max_seq_length)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers, norm=self.bn1)
        # 72 bins for heading, 1 output for altitude, 1 output for speed
        self.linear = Linear(d_model, HDG_BINS + 1 + 1)

    def forward(self, x, attention_mask):
        attention_mask = 1 - attention_mask
        h = self.input_proj(x)
        h = self.encoder(h, src_key_padding_mask=attention_mask)
        h = self.linear(h)

        return h

    @property
    def name(self):
        return f"encoder-{self.d_model}-{self.n_head}-max{self.max_seq_len}-{self.num_layers}_linear1"