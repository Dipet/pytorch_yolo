from torch import nn
from collections import OrderedDict

from pytorch_yolo.models.layers.common import Add


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size=3, stride=1, pad=True):
        super().__init__()

        conv_pad = (size - 1) // 2 if pad else 0

        self.sequence = nn.Sequential(
            OrderedDict(
                [
                    [
                        "conv",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=size,
                            stride=stride,
                            padding=conv_pad,
                            bias=False,
                        ),
                    ],
                    ["batch_norm", nn.BatchNorm2d(out_channels)],
                    ["activation", nn.LeakyReLU(0.1, inplace=True)],
                ]
            )
        )

        self.out_channels = out_channels

    def forward(self, x):
        return self.sequence(x)


class MaxPool(nn.MaxPool2d):
    def __init__(self, size=2, stride=2):
        if size == 2 and stride == 1:
            super().__init__(size, stride, padding=1, dilation=2)
        else:
            pad = (size - 1) // 2
            super().__init__(size, stride, padding=pad)


class ConvMaxPool(nn.Module):
    def __init__(
        self, in_channels, out_channels, conv_size=3, conv_stride=1, conv_pad=True, pool_size=2, pool_stride=2
    ):
        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels, conv_size, conv_stride, conv_pad)
        self.pool = MaxPool(pool_size, pool_stride)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, repeat=1):
        super().__init__()

        self._out_channels = out_channels
        self.conv0 = ConvBlock(in_channels, out_channels, size=3, stride=2)

        self.tail = []
        self.adds = []
        for i in range(repeat):
            name = f"seq{i}"
            name_add = f"add{i}"
            setattr(
                self,
                name,
                nn.Sequential(
                    ConvBlock(out_channels, out_channels // 2, size=1, stride=1),
                    ConvBlock(out_channels // 2, out_channels, size=3, stride=1),
                ),
            )
            setattr(self, name_add, Add())
            self.tail.append(getattr(self, name))
            self.adds.append(getattr(self, name_add))

    def forward(self, x):
        x = self.conv0(x)

        for module, add in zip(self.tail, self.adds):
            sub = module(x)
            x = add(x, sub)

        return x

    @property
    def out_channels(self):
        return self._out_channels
