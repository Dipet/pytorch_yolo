from torch import nn


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, load_weights=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.load_weights = load_weights
