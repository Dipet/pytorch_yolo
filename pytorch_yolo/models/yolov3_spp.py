import torch
from torch import nn

try:
    from yolo_base import YOLOBase, ConvBlock, MaxPool
    from yolo_layer import Concat, Upsample, YOLOLayer
except:
    from .yolo_base import YOLOBase, ConvBlock, MaxPool
    from .yolo_layer import Concat, Upsample, YOLOLayer


class Add(nn.Module):
    def forward(self, x, y):
        return x + y


class DownSample(nn.Module):
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
        sub = x

        for module, add in zip(self.tail, self.adds):
            sub = module(x)
            x = add(x, sub)

        return x, sub

    @property
    def out_channels(self):
        return self._out_channels


class YOLOv3SPP(YOLOBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        kd = self.kernels_divider

        # Darknet Encoder
        # ======================================================================
        self.conv1 = ConvBlock(self.in_channels, 32 // kd, stride=1, size=3)

        self.down1 = DownSample(self.conv1.out_channels, 64 // kd, repeat=0)
        self.down2 = DownSample(self.down1.out_channels, 128 // kd, repeat=1)
        self.down3 = DownSample(self.down2.out_channels, 256 // kd, repeat=7)
        self.down4 = DownSample(self.down3.out_channels, 512 // kd, repeat=7)
        self.down5 = DownSample(self.down4.out_channels, 1024 // kd, repeat=3)
        self.down = [self.down1, self.down2, self.down3, self.down4, self.down5]

        self.sequence_spp = nn.Sequential()
        self.sequence_spp.add_module("conv1", ConvBlock(self.down[-1].out_channels, 512 // kd, size=1, stride=1))
        self.sequence_spp.add_module(
            "conv2", ConvBlock(self.sequence_spp[-1].out_channels, 1024 // kd, size=3, stride=1)
        )
        self.sequence_spp.add_module(
            "conv3", ConvBlock(self.sequence_spp[-1].out_channels, 512 // kd, size=1, stride=1)
        )

        self.spp1 = MaxPool(size=5, stride=1)
        self.spp2 = MaxPool(size=9, stride=1)
        self.spp3 = MaxPool(size=13, stride=1)

        self.branch1_1 = nn.Sequential()
        self.branch1_1.add_module("concat", Concat(1))
        self.branch1_1.add_module(
            "conv1", ConvBlock(self.sequence_spp[-1].out_channels * 4, 512 // kd, size=1, stride=1)
        )
        self.branch1_1.add_module("conv2", ConvBlock(self.branch1_1[-1].out_channels, 1024 // kd, size=3, stride=1))
        self.branch1_1.add_module("conv3", ConvBlock(self.branch1_1[-1].out_channels, 512 // kd, size=1, stride=1))
        self.branch1_2 = nn.Sequential()
        self.branch1_2.add_module("conv1", ConvBlock(self.branch1_1[-1].out_channels, 1024 // kd, size=3, stride=1))
        self.branch1_2.add_module(
            "conv2", ConvBlock(self.branch1_2[-1].out_channels, self.yolo_layer_input_size, size=1, stride=1)
        )

        self.branch2_1 = nn.Sequential(
            ConvBlock(self.branch1_1[-1].out_channels, 256 // kd, size=1, stride=1), Upsample(2)
        )
        self.branch2_2 = nn.Sequential()
        self.branch2_2.add_module("concat", Concat(1))
        self.branch2_2.add_module(
            "conv1", ConvBlock(self.branch2_1[0].out_channels + self.down[3].out_channels, 256 // kd, size=1, stride=1)
        )
        self.branch2_2.add_module("conv2", ConvBlock(self.branch2_2[-1].out_channels, 512 // kd, size=3, stride=1))
        self.branch2_2.add_module("conv3", ConvBlock(self.branch2_2[-1].out_channels, 256 // kd, size=1, stride=1))
        self.branch2_2.add_module("conv4", ConvBlock(self.branch2_2[-1].out_channels, 512 // kd, size=3, stride=1))
        self.branch2_2.add_module("conv5", ConvBlock(self.branch2_2[-1].out_channels, 256 // kd, size=1, stride=1))
        self.branch2_3 = nn.Sequential()
        self.branch2_3.add_module("conv6", ConvBlock(self.branch2_2[-1].out_channels, 512 // kd, size=3, stride=1))
        self.branch2_3.add_module(
            "conv7", ConvBlock(self.branch2_3[-1].out_channels, self.yolo_layer_input_size, size=1, stride=1)
        )

        self.branch3_1 = nn.Sequential(
            ConvBlock(self.branch2_2[-1].out_channels, 128 // kd, size=1, stride=1), Upsample(2)
        )
        self.branch3_2 = nn.Sequential()
        self.branch3_2.add_module("concat", Concat(1))
        self.branch3_2.add_module(
            "conv1", ConvBlock(self.branch3_1[0].out_channels + self.down[2].out_channels, 128 // kd, size=1, stride=1)
        )
        self.branch3_2.add_module("conv2", ConvBlock(self.branch3_2[-1].out_channels, 256 // kd, size=3, stride=1))
        self.branch3_2.add_module("conv3", ConvBlock(self.branch3_2[-1].out_channels, 128 // kd, size=1, stride=1))
        self.branch3_2.add_module("conv4", ConvBlock(self.branch3_2[-1].out_channels, 256 // kd, size=3, stride=1))
        self.branch3_2.add_module("conv5", ConvBlock(self.branch3_2[-1].out_channels, 128 // kd, size=1, stride=1))
        self.branch3_2.add_module("conv6", ConvBlock(self.branch3_2[-1].out_channels, 256 // kd, size=3, stride=1))
        self.branch3_2.add_module(
            "conv7", ConvBlock(self.branch3_2[-1].out_channels, self.yolo_layer_input_size, size=1, stride=1)
        )
        # ======================================================================

        # YOLO Layers
        # ======================================================================
        self.yolo1, self.yolo2, self.yolo3 = self._create_yolo_layers()
        # ======================================================================

    def _forward_encoder(self, x):
        x = self.conv1(x)

        downs = []
        for module in self.down:
            x, sub = module(x)
            downs.append(sub)

        x = self.sequence_spp(x)

        x = self.branch1_1([self.spp1(x), self.spp2(x), self.spp3(x), x])
        branch1 = self.branch1_2(x)

        x = self.branch2_1(x)
        x = self.branch2_2([x, downs[3]])
        branch2 = self.branch2_3(x)

        x = self.branch3_1(x)
        branch3 = self.branch3_2([x, downs[2]])

        return branch1, branch2, branch3

    def forward(self, x):
        img_size = max(x.shape[-2:])

        # Encoder
        # ======================================================================
        branch1, branch2, branch3 = self._forward_encoder(x)
        # ======================================================================

        # YOLO
        # ======================================================================
        branch1 = self.yolo1(branch1, img_size)
        branch2 = self.yolo2(branch2, img_size)
        branch3 = self.yolo3(branch3, img_size)
        # ======================================================================

        out = [branch1, branch2, branch3]
        if self.training:
            return out
        elif self.onnx:
            output = torch.cat(out, 1)
            return output[5:].t(), output[:4].t()

        io, p = list(zip(*out))  # inference output, training output
        return torch.cat(io, 1), p

    @property
    def yolo_layers(self):
        return self.yolo1, self.yolo2, self.yolo3

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

    def fuse(self):
        for layer in self.modules():
            if isinstance(layer, ConvBlock):
                layer.fuse()


if __name__ == "__main__":
    from tensorboardX import SummaryWriter
    from torch import onnx
    from torchsummary import summary

    device = "cuda"
    model = YOLOv3SPP(
        n_class=80,
        onnx=False,
        in_shape=(1, 3, 320, 320),
        anchors=[[(10, 13), (16, 30), (33, 23)], [(30, 61), (62, 45), (59, 119)], [(116, 90), (156, 198), (373, 326)]],
    ).to(device)
    dummy = torch.rand((1, 3, 320, 320)).to(device)
    summary(model, input_size=(3, 608, 608))
    # onnx.export(model, dummy, 'test.onnx')
    # model(dummy)
    #
    # try:
    #     writer.add_graph(model, dummy)
    # except:
    #     writer.add_graph(model, dummy)
