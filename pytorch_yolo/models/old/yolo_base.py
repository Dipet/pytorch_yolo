import torch
import torch.nn.functional as F

from torch import nn

from collections import OrderedDict

from pytorch_yolo.utils.torch_utils import fuse_conv_and_bn
from pytorch_yolo.models.old.yolo_layer import YOLOLayer


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size=3, stride=1, pad=True):
        super().__init__()

        conv_pad = (size - 1) // 2 if pad else 0

        self.sequence = nn.Sequential(
            OrderedDict(
                (
                    (
                        "conv",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=size,
                            stride=stride,
                            padding=conv_pad,
                            bias=False,
                        ),
                    ),
                    ("batch_norm", nn.BatchNorm2d(out_channels)),
                    ("activation", nn.LeakyReLU(0.1, inplace=True)),
                )
            )
        )

        self.out_channels = out_channels

    def forward(self, x):
        return self.sequence(x)

    def fuse(self):
        fused_sequence = []
        for i, layer in enumerate(self.sequence):
            if isinstance(layer, nn.BatchNorm2d):
                # fuse this bn layer with the previous conv2d layer
                conv = self.sequence[i - 1]
                layer = fuse_conv_and_bn(conv, layer)
                fused_sequence = fused_sequence[:-1] + [layer]
                continue

            fused_sequence.append(layer)
        self.sequence = nn.Sequential(*fused_sequence)


class MaxPool(nn.MaxPool2d):
    def __init__(self, size, stride):
        if size == 2 and stride == 1:
            super().__init__(size, stride, padding=1, dilation=2)
        else:
            pad = (size - 1) // 2
            super().__init__(size, stride, padding=pad)


class ConvPoolBlock(ConvBlock):
    def __init__(
            self, in_channels, out_channels, conv_size=3, conv_stride=1, conv_pad=True, pool_size=2, pool_stride=2
    ):
        super().__init__(in_channels, out_channels, conv_size, conv_stride, conv_pad)
        self.sequence.add_module("max_pool", MaxPool(pool_size, pool_stride))


class Upsample(nn.Module):
    def __init__(self, scale_factor=1, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class Concat(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, args):
        return torch.cat(args, self.dim)


class YOLOBase(nn.Module):
    def __init__(
            self,
            in_channels=3,
            num_class=80,
            anchors=(((10.0, 14.0), (23.0, 27.0), (37.0, 58.0)), ((81.0, 82.0), (135.0, 169.0), (344.0, 319.0))),
            activation=None,
    ):
        super().__init__()

        self.num_class = num_class
        self.in_channels = in_channels
        self.anchors = anchors
        self.activation = activation

        self.yolo_layers = self._create_yolo_layers()

    def _create_yolo_layers(self):
        layers = []

        for anchor in self.anchors:
            layers.append(YOLOLayer(anchor, self.num_class, class_activation=self.activation))

        return layers