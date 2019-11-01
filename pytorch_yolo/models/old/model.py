from torch import nn


class Model(nn.Module):
    def __init__(self, in_channels, num_classes, load_weights, anchors):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.load_weights = load_weights
        self.anchors = anchors
