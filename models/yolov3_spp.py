import torch
from torch import nn

try:
    from yolo_base import YOLOBase, ConvBlock, ConvPoolBlock
    from yolo_layer import Concat, Upsample
except:
    from .yolo_base import YOLOBase, ConvBlock, ConvPoolBlock
    from .yolo_layer import Concat, Upsample


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, sizes, strides,
                 continue_prev=False):
        super().__init__()

        assert len(sizes) == len(strides)

        self.conv0 = ConvBlock(in_channels, out_channels[0],
                               size=sizes[0], stride=strides[0])
        self.sequence = nn.Sequential()

        prev = self.conv1
        for i in range(1, len(sizes)):
            module = ConvBlock(prev.out_channels, out_channels[i],
                               size=sizes[i], stride=strides[i])

            self.sequence.add_module(f'conv{i}', module)
            prev = module

    def forward(self, x):
        sub = self.conv0(x)

        return sub + self.sequence(x)

    @property
    def out_channels(self):
        return self.conv0.out_channels


class YOLOv3SPP(YOLOBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Darknet Encoder
        # ======================================================================
        self.conv1 = ConvBlock(self.in_channels, 32, stride=1, size=3)

        self.down1 = DownSample(self.conv1.out_channels,
                                out_channels=[64, 32, 64],
                                sizes=[3, 1, 3],
                                strides=[2, 1, 1])

        self.down2_1 = DownSample(self.down1.out_channels,
                                  out_channels=[128, 64, 128],
                                  sizes=[3, 1, 3],
                                  strides=[2, 1, 1])
        self.down2_2 = DownSample(self.down2_1.out_channels,
                                  out_channels=[64, 128],
                                  sizes=[3, 1, 3],
                                  strides=[2, 1, 1])
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

    device = 'cuda'
    model = YOLOv3SPP(n_class=1, onnx=False, in_shape=(1, 3, 320, 320)).to(device).eval()
    dummy = torch.rand((1, 3, 320, 320)).to(device)
    # onnx.export(model, dummy, 'test.onnx')
    # model(dummy)
    #
    # try:
    #     writer.add_graph(model, dummy)
    # except:
    #     writer.add_graph(model, dummy)
