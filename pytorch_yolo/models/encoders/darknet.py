from torch import nn

from .encoder import Encoder
from pytorch_yolo.models.layers.darknet_layers import ConvBlock, MaxPool, ConvMaxPool, ResBlock


class DarknetTinyV3(Encoder):
    def __init__(self, in_channels=3, conv_channels=(16, 32, 64, 128, 256, 512, 1024), load_weights=False):
        super().__init__(in_channels, conv_channels, load_weights)

        assert len(conv_channels) == 7, "DarknetTinyV3 contains 7 conv layers"

        self.sequence = nn.Sequential(
            ConvMaxPool(self.in_channels, conv_channels[0]),
            ConvMaxPool(conv_channels[0], conv_channels[1]),
            ConvMaxPool(conv_channels[1], conv_channels[2]),
            ConvMaxPool(conv_channels[2], conv_channels[3]),
            ConvBlock(conv_channels[3], conv_channels[4]),
            MaxPool(),
            nn.Sequential(
                ConvMaxPool(conv_channels[4], conv_channels[5], pool_stride=1),
                ConvBlock(conv_channels[5], conv_channels[6]),
            ),
        )

        num_channels = len(conv_channels)
        self.outputs = [num_channels - 3, num_channels - 1]
        self.out_channels = [conv_channels[i] for i in self.outputs]

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.sequence):
            x = layer(x)

            if i in self.outputs:
                outputs.append(x)

        return outputs

    @staticmethod
    def _load_weights(sequence, weights):
        for layer in sequence:
            if isinstance(layer, (ConvBlock, ConvMaxPool)):
                weights = layer.load_darknet_weights(weights)
            elif isinstance(layer, nn.Sequential):
                weights = DarknetTinyV3._load_weights(layer, weights)

        return weights

    def load_darknet_weights(self, weights):
        return self._load_weights(self.sequence, weights)


class DarknetV3(Encoder):
    def __init__(
        self,
        in_channels=3,
        conv_channels=(32, 64, 128, 256, 512, 1024),
        res_blocks=(1, 2, 8, 8, 4),
        load_weights=False,
    ):
        super().__init__(in_channels, conv_channels, load_weights)

        assert len(conv_channels) == 6, "DarknetV3 contains 6 conv layers"
        assert len(res_blocks) == 5, "DarknetV3 contains 5 res blocks"

        self.sequence = nn.Sequential(
            ConvBlock(in_channels, conv_channels[0]),
            ResBlock(conv_channels[0], conv_channels[1], res_blocks[0]),
            ResBlock(conv_channels[1], conv_channels[2], res_blocks[1]),
            ResBlock(conv_channels[2], conv_channels[3], res_blocks[2]),
            ResBlock(conv_channels[3], conv_channels[4], res_blocks[3]),
            ResBlock(conv_channels[4], conv_channels[5], res_blocks[4]),
        )

        num_channels = len(conv_channels)
        self.outputs = [num_channels - 3, num_channels - 2, num_channels - 1]
        self.out_channels = [conv_channels[i] for i in self.outputs]

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.sequence):
            x = layer(x)
            if i in self.outputs:
                outputs.append(x)

        return outputs

    def load_darknet_weights(self, weights):
        for layer in self.sequence:
            weights = layer.load_darknet_weights(weights)

        return weights
