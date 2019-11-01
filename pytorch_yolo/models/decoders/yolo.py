from torch import nn

from pytorch_yolo.models.layers.darknet_layers import ConvBlock, MaxPool
from pytorch_yolo.models.layers.common import Concat, Upsample
from pytorch_yolo.models.layers.yolo_layer import YoloLayer


class TinyV3Decoder(nn.Module):
    def __init__(
        self,
        in_channels=(256, 1024),
        num_classes=80,
        anchors=(((10, 13), (16, 30), (33, 23)), ((30, 61), (62, 45), (59, 119)), ((116, 90), (156, 198), (373, 326))),
        channels=(128, 256),
        activation=None,
    ):
        super().__init__()
        self.anchosrs = anchors
        self.num_classes = num_classes
        self.channels = channels

        assert len(anchors) == 2, "Tiny YOLO supports only 2 YOLO layers!"
        assert len(channels) == 2, "Tiny YOLO supports only 2 YOLO layers!"
        self.yolo1 = YoloLayer(anchors[0], num_classes, activation)
        self.yolo2 = YoloLayer(anchors[1], num_classes, activation)

        self.head1 = nn.Sequential(
            ConvBlock(in_channels[0] + channels[0], 2 * channels[0]),
            nn.Conv2d(2 * channels[0], self.yolo1.input_channels, 1, 1),
        )

        self.sub_head2 = ConvBlock(in_channels[1], channels[1], 1)
        self.head2 = nn.Sequential(
            ConvBlock(channels[1], 2 * channels[1]), nn.Conv2d(2 * channels[1], self.yolo2.input_channels, 1, 1)
        )

        self.up = nn.Sequential(ConvBlock(channels[1], channels[0], 1), Upsample(2))

        self.concat = Concat(1)

    def forward(self, x, image_shape):
        image_size = max(image_shape)

        x1, x2 = x

        x = self.sub_head2(x2)
        x2 = self.head2(x)
        y2 = self.yolo2(x2, image_size)

        x = self.up(x)
        x1 = self.concat([x, x1])
        x1 = self.head1(x1)
        y1 = self.yolo1(x1, image_size)

        return y1, y2


class YoloV3Decoder(nn.Module):
    def __init__(
        self,
        in_channels=(256, 512, 1024),
        num_classes=80,
        anchors=(((10, 13), (16, 30), (33, 23)), ((30, 61), (62, 45), (59, 119)), ((116, 90), (156, 198), (373, 326))),
        channels=(128, 256, 512),
        activation=None,
    ):
        super().__init__()

        self.anchosrs = anchors
        self.num_classes = num_classes
        self.channels = channels

        assert len(anchors) == 3, "YOLOv3 supports only 3 YOLO layers!"
        assert len(channels) == 3, "YOLOv3 supports only 3 YOLO layers!"
        self.yolo1, self.yolo2, self.yolo3 = [YoloLayer(i, num_classes, activation) for i in anchors]

        self.sub_head1, self.head1 = self._make_last_layers(
            in_channels[0] + channels[0], channels[0], self.yolo1.input_channels
        )
        self.sub_head2, self.head2 = self._make_last_layers(
            in_channels[1] + channels[1], channels[1], self.yolo2.input_channels
        )
        self.sub_head3, self.head3 = self._make_last_layers(in_channels[2], channels[2], self.yolo3.input_channels)

        self.concat = Concat(dim=1)
        self.up2 = nn.Sequential(ConvBlock(channels[2], channels[1], 1), Upsample(2))
        self.up1 = nn.Sequential(ConvBlock(channels[1], channels[0], 1), Upsample(2))

    def _make_last_layers(self, in_channels, num_channels, yolo_input_channels):
        double_channels = num_channels * 2

        x = nn.Sequential(
            ConvBlock(in_channels, num_channels, 1),
            ConvBlock(num_channels, double_channels, 3),
            ConvBlock(double_channels, num_channels, 1),
            ConvBlock(num_channels, double_channels, 3),
            ConvBlock(double_channels, num_channels, 1),
        )

        y = nn.Sequential(
            ConvBlock(num_channels, double_channels, 3),
            nn.Conv2d(double_channels, yolo_input_channels, 1, 1),
        )

        return x, y

    def forward(self, x, image_shape):
        img_size = max(image_shape)
        x1, x2, x3 = x

        x = self.sub_head3(x3)
        x3 = self.head3(x)
        y3 = self.yolo3(x3, img_size)

        x = self.up2(x)
        x = self.concat([x, x2])
        x = self.sub_head2(x)
        x2 = self.head2(x)
        y2 = self.yolo2(x2, img_size)

        x = self.up1(x)
        x = self.concat([x, x1])
        x = self.sub_head1(x)
        x1 = self.head1(x)
        y1 = self.yolo1(x1, img_size)

        return y1, y2, y3


class YoloV3SppDecoder(nn.Module):
    def __init__(
            self,
            in_channels=(256, 512, 1024),
            num_classes=80,
            anchors=(
            ((10, 13), (16, 30), (33, 23)), ((30, 61), (62, 45), (59, 119)), ((116, 90), (156, 198), (373, 326))),
            channels=(128, 256, 512),
            activation=None,
    ):
        super().__init__()

        self.anchosrs = anchors
        self.num_classes = num_classes
        self.channels = channels

        assert len(anchors) == 3, "YOLOv3 supports only 3 YOLO layers!"
        assert len(channels) == 3, "YOLOv3 supports only 3 YOLO layers!"
        self.yolo1, self.yolo2, self.yolo3 = [YoloLayer(i, num_classes, activation) for i in anchors]

        self.spp = nn.Sequential(
            ConvBlock(in_channels[-1], channels[-1], 1),
            ConvBlock(channels[-1], 2 * channels[-1], 3),
            ConvBlock(2 * channels[-1], channels[-1], 1)
        )
        self.spp1 = MaxPool(5, 1)
        self.spp2 = MaxPool(9, 1)
        self.spp3 = MaxPool(13, 1)

        self.sub_head1, self.head1 = self._make_last_layers(in_channels[0] + channels[0], channels[0], self.yolo1.input_channels, repeat=1)
        self.sub_head2, self.head2 = self._make_last_layers(in_channels[1] + channels[1], channels[1], self.yolo1.input_channels, repeat=2)
        self.sub_head3, self.head3 = self._make_last_layers(channels[0] * 4, channels[2], self.yolo1.input_channels, repeat=2)

        self.concat = Concat(dim=1)
        self.up2 = nn.Sequential(ConvBlock(channels[2], channels[1], 1), Upsample(2))
        self.up1 = nn.Sequential(ConvBlock(channels[1], channels[0], 1), Upsample(2))

    def _make_last_layers(self, in_channels, num_channels, yolo_input_channels, repeat=1):
        double_channels = num_channels * 2

        layers = [ConvBlock(in_channels, num_channels, 1)]
        for i in range(repeat):
            layers.append(ConvBlock(num_channels, double_channels, 3))
            layers.append(ConvBlock(double_channels, num_channels, 1))

        x = nn.Sequential(*layers)

        y = nn.Sequential(
            ConvBlock(num_channels, double_channels, 3),
            nn.Conv2d(double_channels, yolo_input_channels, 1, 1),
        )

        return x, y

    def forward(self, x):
        x1, x2, x3 = x

        x = self.spp(x3)
        x = self.concat([x, self.spp1(x), self.spp2(x), self.spp3(x)])

        x = self.sub_head3(x)
        x3 = self.head3(x)
        y3 = self.yolo3(x3)

        x = self.up2(x)
        self.concat([x, x2])
        x = self.sub_head2(x)
        x2 = self.head2(x)
        y2 = self.yolo2(x2)

        x = self.up1(x)
        self.concat([x, x1])
        x = self.sub_head1(x)
        x1 = self.head1(x)
        y1 = self.yolo1(x1)

        return y1, y2, y3
