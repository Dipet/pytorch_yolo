from torch import nn

from models.layers.darknet_layers import ConvBlock
from models.layers.common import Concat, Upsample


class YOLOLayer(nn.Module):
    def __init__(self, num_classes, anchors):
        super().__init__()

        self.anchors = anchors
        self.num_classes = num_classes

    def forward(self, p, img_size):
        batch_size, num_channels, height, width = p.shape

        if (self.n_x_grids, self.n_y_grids) != (height, width):
            self.create_grids(p.device)

        p = (
            p.view(batch_size, len(self.anchors), self.num_classes + 5, height, width)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        return p

    @property
    def input_channels(self):
        return (self.num_classes + 5) * len(self.anchors)


class TinyV3Decoder(nn.Module):
    def __init__(
        self,
        in_channels=(256, 1024),
        num_classes=80,
        anchors=(((10, 14), (23, 27), (37, 58)), ((81, 82), (135, 169), (344, 319))),
        channels=((256, 512), (128, 256)),
    ):
        super().__init__()
        self.anchosrs = anchors
        self.num_classes = num_classes
        self.channels = channels

        assert len(anchors) == 2, "Tiny YOLO supports only 2 YOLO layers!"
        assert len(channels) == 2, "Tiny YOLO supports only 2 YOLO layers!"
        self.yolo1 = YOLOLayer(num_classes, anchors[0])
        self.yolo2 = YOLOLayer(num_classes, anchors[1])

        self.conv1 = self._gen_yolo_conv([in_channels[0]] + list(channels[0]), self.yolo1)
        self.conv2 = self._gen_yolo_conv([in_channels[1]] + list(channels[1]), self.yolo2)
        self.concat = Concat(1)
        self.upsample = Upsample(2)

    def _gen_yolo_conv(self, channels, yolo_layer: YOLOLayer):
        assert len(channels) >= 1, "Empty channels for YOLO layer."

        layers = nn.Sequential()

        for i, (ic1, oc1) in enumerate(zip(channels[:-1], channels[1:])):
            if i % 2 == 0:
                layers.add_module(f"conv{i}", ConvBlock(ic1, oc1, 1))
            else:
                layers.add_module(f"conv{i}", ConvBlock(ic1, oc1, 3))

        layers.add_module("conv_last", ConvBlock(channels[-1], yolo_layer.input_channels, 1))

        return layers

    def forward(self, x1, x2):
        x = self.conv1[0](x1)

        x1 = x
        for layer in self.conv1[1:]:
            x1 = layer(x1)

        x = self.upsample(self.conv2[0](x))
        x2 = self.concat([x, x2])
        for layer in self.conv2[1:]:
            x2 = layer(x2)

        x1, x2 = x2, x1

        return self.yolo1(x1), self.yolo2(x2)
