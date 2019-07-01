import torch
from torch import nn

try:
    from yolo_base import YOLOBase, ConvBlock, ConvPoolBlock
    from yolo_layer import Concat, Upsample
except:
    from .yolo_base import YOLOBase, ConvBlock, ConvPoolBlock
    from .yolo_layer import Concat, Upsample


class YOLOv3Tiny(YOLOBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Darknet Encoder
        # ======================================================================
        self.sequence_1 = nn.Sequential()
        self.sequence_1.add_module('conv1', ConvPoolBlock(self.in_channels, max(8, 16 // self.kernels_divider)))
        self.sequence_1.add_module('conv2', ConvPoolBlock(self.sequence_1[-1].out_channels, max(8, 32 //self.kernels_divider)))
        self.sequence_1.add_module('conv3', ConvPoolBlock(self.sequence_1[-1].out_channels, max(8, 64 //self.kernels_divider)))
        self.sequence_1.add_module('conv4', ConvPoolBlock(self.sequence_1[-1].out_channels, max(8, 128 //self.kernels_divider)))
        self.sequence_1.add_module('conv5', ConvBlock(self.sequence_1[-1].out_channels, max(8, 256 //self.kernels_divider)))

        self.sequence_2 = nn.Sequential()
        self.sequence_2.add_module('max_pool5', nn.MaxPool2d(2, 2))
        self.sequence_2.add_module('conv6', ConvPoolBlock(self.sequence_1[-1].out_channels, max(8, 512 //self.kernels_divider), pool_stride=1))
        self.sequence_2.add_module('conv7', ConvBlock(self.sequence_2[-1].out_channels, max(8, 1024 //self.kernels_divider)))
        self.sequence_2.add_module('conv8', ConvBlock(self.sequence_2[-1].out_channels, max(8, 256 //self.kernels_divider), size=1))

        self.sequence_branch1_1 = nn.Sequential()
        self.sequence_branch1_1.add_module('branch1_conv1', ConvBlock(self.sequence_2[-1].out_channels, max(8, 128 //self.kernels_divider), size=1))
        self.sequence_branch1_1.add_module('branch1_upsample', Upsample(2))

        self.sequence_branch1_2 = nn.Sequential()
        self.sequence_branch1_2.add_module('branch1_concat', Concat(1))
        self.sequence_branch1_2.add_module('branch1_conv2', ConvBlock(self.sequence_1[-1].out_channels + self.sequence_branch1_1[0].out_channels, max(8, 256 //self.kernels_divider)))
        self.sequence_branch1_2.add_module('branch1_conv3', nn.Conv2d(self.sequence_branch1_2[-1].out_channels, self.yolo_layer_input_size, kernel_size=1))

        self.sequence_branch2 = nn.Sequential()
        self.sequence_branch2.add_module('branch2_conv1', ConvBlock(self.sequence_2[-1].out_channels, max(8, 512 //self.kernels_divider)))
        self.sequence_branch2.add_module('branch2_conv2', nn.Conv2d(self.sequence_branch2[-1].out_channels, self.yolo_layer_input_size, kernel_size=1))
        # ======================================================================

        # YOLO Layers
        # ======================================================================
        self.yolo1, self.yolo2 = self._create_yolo_layers()
        # ======================================================================

    @property
    def yolo_layers(self):
        return self.yolo1, self.yolo2

    def load_darknet_weights(self, weights_path, warnings=True):
        layers = list(self.sequence_1) + list(self.sequence_2)
        self._load_darknet_weights(weights_path, layers, warnings)

    def save_darknet_weights(self, path, warnings=True):
        layers = list(self.sequence_1) + list(self.sequence_2)
        # Ветки поменялись местами, т.к. они в оригинале определяются
        # в таком порядке
        layers += list(self.sequence_branch2)
        layers += list(self.sequence_branch1_1) + list(self.sequence_branch1_2)

        self._save_weights(path, layers, warnings)

    def _forward_encoder(self, x):
        x = self.sequence_1(x)
        x_route1 = x
        x_route2 = self.sequence_2(x)

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

    def fuse(self):
        for layer in self.modules():
            if isinstance(layer, ConvBlock):
                layer.fuse()


if __name__ == '__main__':
    from tensorboardX import SummaryWriter
    from torch import onnx
    from torchsummary import summary

    device = 'cpu'
    model = YOLOv3Tiny(n_class=1, onnx=False, in_shape=(1, 3, 320, 320)).to(device).eval()
    dummy = torch.rand((1, 3, 320, 320)).to(device)
    summary(model, (3, 608, 608), device=device)
    # onnx.export(model, dummy, 'test.onnx')
    # model(dummy)
    #
    # try:
    #     writer.add_graph(model, dummy)
    # except:
    #     writer.add_graph(model, dummy)
