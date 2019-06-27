import torch
from torch import nn
from torchvision.models import shufflenet_v2_x1_0

try:
    from yolo_base import YOLOBase, ConvBlock, ConvPoolBlock
    from yolo_layer import YOLOLayer, Concat, Upsample
except:
    from .yolo_base import YOLOBase, ConvBlock, ConvPoolBlock
    from .yolo_layer import YOLOLayer, Concat, Upsample


class ShuffleEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        model = shufflenet_v2_x1_0(True)

        if in_channels == 3:
            conv1 = model.conv1
        else:
            output_channels = list(model.conv1.modules())[-2].num_feauters
            conv1 = nn.Sequential(
                nn.Conv2d(in_channels, output_channels, 3, 2, 1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        self.sequence1 = nn.Sequential(conv1,
                                       model.maxpool,
                                       model.stage2,
                                       model.stage3)

        self.sequence2 = nn.Sequential(model.stage4,
                                       model.conv5)

    def forward(self, x):
        b1 = self.sequence1(x)
        b2 = self.sequence2(b1)
        return b1, b2

    @property
    def out_channels(self):
        out1 = list(self.sequence1.modules())[-2].num_features * 2
        out2 = list(self.sequence2.modules())[-2].num_features

        return out1, out2


class YOLOv3TinyShuffle(YOLOBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Darknet Encoder
        # ======================================================================
        self.features = ShuffleEncoder(in_channels=self.in_channels)
        f_out1, f_out2 = self.features.out_channels


        self.sequence_branch1_1 = nn.Sequential()
        self.sequence_branch1_1.add_module('branch1_conv1', ConvBlock(f_out2, max(8, 128 //  self.kernels_divider), size=1))
        self.sequence_branch1_1.add_module('branch1_upsample', Upsample(2))

        self.sequence_branch1_2 = nn.Sequential()
        self.sequence_branch1_2.add_module('branch1_concat', Concat(1))
        self.sequence_branch1_2.add_module('branch1_conv2', ConvBlock(f_out1 + self.sequence_branch1_1[0].out_channels, max(8, 128 //  self.kernels_divider)))
        self.sequence_branch1_2.add_module('branch1_conv3', nn.Conv2d(self.sequence_branch1_2[-1].out_channels, self.yolo_layer_input_size, kernel_size=1))

        self.sequence_branch2 = nn.Sequential()
        self.sequence_branch2.add_module('branch2_conv1', ConvBlock(f_out2, max(8, 128 //  self.kernels_divider)))
        self.sequence_branch2.add_module('branch2_conv2', nn.Conv2d(self.sequence_branch2[-1].out_channels, self.yolo_layer_input_size, kernel_size=1))
        # ======================================================================

        self.yolo1, self.yolo2 = self._create_yolo_layers()

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


if __name__ == '__main__':
    from torchsummary import summary

    device = 'cpu'
    model = YOLOv3TinyShuffle(n_class=1).to(device)
    summary(model, (3, 608, 608), device=device)

    model = ShuffleEncoder()
    b1, b2 = model(torch.rand((1, 3, 608, 608)))
    print(b1.shape, b2.shape)

    print(model.out_channels)
