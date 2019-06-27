import torch
from torch import nn

try:
    from yolo_base import YOLOBase, ConvBlock, ConvPoolBlock
    from yolo_layer import YOLOLayer, Concat, Upsample
except:
    from .yolo_base import YOLOBase, ConvBlock, ConvPoolBlock
    from .yolo_layer import YOLOLayer, Concat, Upsample

from torchvision.models.mobilenet import mobilenet_v2, ConvBNReLU


class MobileNetEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        model = mobilenet_v2(pretrained=True)
        modules = list(model.features)

        self.route_index = 14

        if in_channels == 3:
            self.sequence1 = nn.Sequential(*modules[:self.route_index])
        else:
            conv = ConvBNReLU(in_channels, 32, stride=2)
            nn.init.kaiming_normal_(conv.weight, mode='fan_out')
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

            layers = [conv] + modules[1:self.route_index]
            self.sequence1 = nn.Sequential(*layers)

        self.sequence2 = nn.Sequential(*modules[self.route_index:])

    def forward(self, x):
        branch1 = self.sequence1(x)
        branch2 = self.sequence2(branch1)

        return branch1, branch2

    @property
    def out_channels(self):
        out1 = list(self.sequence1[-1].modules())[-1].num_features
        out2 = list(self.sequence2[-1].modules())[-2].num_features
        return out1, out2


class YOLOv3TinyMobile(YOLOBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Darknet Encoder
        # ======================================================================
        self.features = MobileNetEncoder(in_channels=self.in_channels)
        f_out1, f_out2 = self.features.out_channels

        self.sequence_branch1_1 = nn.Sequential()
        self.sequence_branch1_1.add_module('branch1_conv1', ConvBlock(f_out2, max(8, 128 //  self.kernels_divider), size=1))
        self.sequence_branch1_1.add_module('branch1_upsample', Upsample(2))

        self.sequence_branch1_2 = nn.Sequential()
        self.sequence_branch1_2.add_module('branch1_concat', Concat(1))
        self.sequence_branch1_2.add_module('branch1_conv2', ConvBlock(f_out1 + self.sequence_branch1_1[0].out_channels, max(8, 64 //  self.kernels_divider)))
        self.sequence_branch1_2.add_module('branch1_conv3', nn.Conv2d(self.sequence_branch1_2[-1].out_channels, self.yolo_layer_input_size, kernel_size=1))

        self.sequence_branch2 = nn.Sequential()
        self.sequence_branch2.add_module('branch2_conv1', ConvBlock(f_out2, max(8, 64 //  self.kernels_divider)))
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
    model = YOLOv3TinyMobile(n_class=1).to(device)
    summary(model, (3, 608, 608), device=device)
