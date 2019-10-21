import torch
from torch import nn
from efficientnet_pytorch import EfficientNet

try:
    from yolo_base import YOLOBase, ConvBlock, ConvPoolBlock
    from yolo_layer import YOLOLayer, Concat, Upsample
except:
    from .yolo_base import YOLOBase, ConvBlock, ConvPoolBlock
    from .yolo_layer import YOLOLayer, Concat, Upsample


class SwissActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class EfficientEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        model = EfficientNet.from_pretrained("efficientnet-b0")

        if in_channels == 3:
            self.stem = nn.Sequential(model._conv_stem, model._bn0, SwissActivation())
        else:
            conv = model._conv_stem
            bn = model._bn0
            self.stem = nn.Sequential(
                model._conv_stem.__class__(
                    in_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, bias=False
                ),
                nn.BatchNorm2d(num_features=bn.num_features, momentum=bn.momentum, eps=bn.eps),
                SwissActivation(),
            )

        self.route = 11
        self.sequence1 = nn.ModuleList(model._blocks[: self.route])
        self.sequence2 = nn.ModuleList(model._blocks[self.route :])

        self.drop_connect_rate = model._global_params.drop_connect_rate

    def _forward_blocks(self, x, blocks, i, len_blocks):
        for block in blocks:
            drop_connect_rate = self.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= i / len_blocks
            x = block(x, drop_connect_rate)
            i += 1

        return x, i

    def forward(self, x):
        x = self.stem(x)

        i = 0
        len_blocks = len(self.sequence1) + len(self.sequence2)

        b1, i = self._forward_blocks(x, self.sequence1, i, len_blocks)
        b2, i = self._forward_blocks(b1, self.sequence2, i, len_blocks)

        return b1, b2

    @property
    def out_channels(self):
        out1 = self.sequence1[-1]._block_args.output_filters
        out2 = self.sequence2[-1]._block_args.output_filters

        return out1, out2


class YOLOv3TinyEfficient(YOLOBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Darknet Encoder
        # ======================================================================
        self.features = EfficientEncoder(in_channels=self.in_channels)
        f_out1, f_out2 = self.features.out_channels

        self.sequence_branch1_1 = nn.Sequential()
        self.sequence_branch1_1.add_module(
            "branch1_conv1", ConvBlock(f_out2, max(8, 128 // self.kernels_divider), size=1)
        )
        self.sequence_branch1_1.add_module("branch1_upsample", Upsample(2))

        self.sequence_branch1_2 = nn.Sequential()
        self.sequence_branch1_2.add_module("branch1_concat", Concat(1))
        self.sequence_branch1_2.add_module(
            "branch1_conv2",
            ConvBlock(f_out1 + self.sequence_branch1_1[0].out_channels, max(8, 128 // self.kernels_divider)),
        )
        self.sequence_branch1_2.add_module(
            "branch1_conv3",
            nn.Conv2d(self.sequence_branch1_2[-1].out_channels, self.yolo_layer_input_size, kernel_size=1),
        )

        self.sequence_branch2 = nn.Sequential()
        self.sequence_branch2.add_module("branch2_conv1", ConvBlock(f_out2, max(8, 128 // self.kernels_divider)))
        self.sequence_branch2.add_module(
            "branch2_conv2",
            nn.Conv2d(self.sequence_branch2[-1].out_channels, self.yolo_layer_input_size, kernel_size=1),
        )
        # ======================================================================

        # YOLO Layers
        # ======================================================================
        self.yolo1, self.yolo2 = self._create_yolo_layers()
        # ======================================================================

    @property
    def yolo_layers(self):
        return self.yolo1, self.yolo2

    def _forward_encoder(self, x):
        x_route1, x_route2 = self.features(x)

        x_branch1 = self.sequence_branch1_1(x_route2)
        x_branch1 = self.sequence_branch1_2([x_route1, x_branch1])

        x_branch2 = self.sequence_branch2(x_route2)

        return x_branch1, x_branch2

    def forward(self, x):
        img_size = max(x.shape[-2:])

        # Encoder
        # ======================================================================
        x_branch1, x_branch2 = self._forward_encoder(x)
        # ======================================================================

        # YOLO
        # ======================================================================
        x_branch1 = self.yolo1(x_branch1, img_size)
        x_branch2 = self.yolo2(x_branch2, img_size)
        # ======================================================================

        if self.training:
            return [x_branch1, x_branch2]
        elif self.onnx:
            output = torch.cat([x_branch1, x_branch2], 1)
            return output[5:].t(), output[:4].t()

        io, p = list(zip(x_branch1, x_branch2))  # inference output, training output
        return torch.cat(io, 1), p


if __name__ == "__main__":
    from torchsummary import summary

    device = "cpu"
    model = YOLOv3TinyEfficient(n_class=1).to(device)
    summary(model, (3, 608, 608), device=device)

    model = EfficientEncoder()
    b1, b2 = model(torch.rand((1, 3, 608, 608)))
    print(b1.shape, b2.shape)

    print(model.out_channels)
