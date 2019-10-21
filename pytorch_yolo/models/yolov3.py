import torch
from torch import nn

try:
    from yolo_base import YOLOBase, ConvBlock, MaxPool
    from yolo_layer import Concat, Upsample, YOLOLayer
    from yolov3_spp import DownSample
except:
    from .yolo_base import YOLOBase, ConvBlock, MaxPool
    from .yolo_layer import Concat, Upsample, YOLOLayer
    from .yolov3_spp import DownSample


class YOLOv3(YOLOBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        kd = self.kernels_divider

        self.conv1 = ConvBlock(self.in_channels, 32 // kd)

        self.down1 = DownSample(self.conv1.out_channels, 64 // kd, repeat=0)
        self.down2 = DownSample(self.down1.out_channels, 128 // kd, repeat=1)
        self.down3 = DownSample(self.down2.out_channels, 256 // kd, repeat=7)
        self.down4 = DownSample(self.down3.out_channels, 512 // kd, repeat=7)
        self.down5 = DownSample(self.down4.out_channels, 1024 // kd, repeat=3)
        self.down = [self.down1, self.down2, self.down3, self.down4, self.down5]

        self.seq = nn.Sequential()
        self.seq.add_module("conv1", ConvBlock(self.down[-1].out_channels, 512 // kd, 1))
        self.seq.add_module("conv2", ConvBlock(self.seq[-1].out_channels, 1024 // kd, 3))
        self.seq.add_module("conv3", ConvBlock(self.seq[-1].out_channels, 512 // kd, 1))
        self.seq.add_module("conv4", ConvBlock(self.seq[-1].out_channels, 1024 // kd, 3))
        self.seq.add_module("conv5", ConvBlock(self.seq[-1].out_channels, 512 // kd, 1))

        self.seq_y1 = nn.Sequential()
        self.seq_y1.add_module("conv1", ConvBlock(self.seq[-1].out_channels, 1024 // kd, 3))
        self.seq_y1.add_module("conv2", nn.Conv2d(self.seq_y1[-1].out_channels, self.yolo_layer_input_size, 1, 1))

        self.seqy2_1 = nn.Sequential()
        self.seqy2_1.add_module("conv", ConvBlock(self.seq[-1].out_channels, 256 // kd, 1))
        self.seqy2_1.add_module("up", Upsample(2))

        self.seqy2_2 = nn.Sequential()
        self.seqy2_2.add_module("concat", Concat(1))
        self.seqy2_2.add_module(
            "conv1", ConvBlock(self.seqy2_1[0].out_channels + self.down4.out_channels, 256 // kd, 1)
        )
        self.seqy2_2.add_module("conv2", ConvBlock(self.seqy2_2[-1].out_channels, 512 // kd))
        self.seqy2_2.add_module("conv3", ConvBlock(self.seqy2_2[-1].out_channels, 256 // kd, 1))
        self.seqy2_2.add_module("conv4", ConvBlock(self.seqy2_2[-1].out_channels, 512 // kd))
        self.seqy2_2.add_module("conv5", ConvBlock(self.seqy2_2[-1].out_channels, 256 // kd, 1))

        self.seqy2_3 = nn.Sequential()
        self.seqy2_3.add_module("conv6", ConvBlock(self.seqy2_2[-1].out_channels, 512 // kd))
        self.seqy2_3.add_module("conv7", nn.Conv2d(self.seqy2_3[-1].out_channels, self.yolo_layer_input_size, 1, 1))

        self.seqy3_1 = nn.Sequential()
        self.seqy3_1.add_module("conv", ConvBlock(self.seqy2_2[-1].out_channels, 128 // kd, 1))
        self.seqy3_1.add_module("up", Upsample(2))

        self.seqy3_2 = nn.Sequential()
        self.seqy3_2.add_module("concat", Concat(1))
        self.seqy3_2.add_module(
            "conv1", ConvBlock(self.seqy3_1[0].out_channels + self.down3.out_channels, 128 // kd, 1)
        )
        self.seqy3_2.add_module("conv2", ConvBlock(self.seqy3_2[-1].out_channels, 256 // kd))
        self.seqy3_2.add_module("conv3", ConvBlock(self.seqy3_2[-1].out_channels, 128 // kd, 1))
        self.seqy3_2.add_module("conv4", ConvBlock(self.seqy3_2[-1].out_channels, 256 // kd))
        self.seqy3_2.add_module("conv5", ConvBlock(self.seqy3_2[-1].out_channels, 128 // kd, 1))
        self.seqy3_2.add_module("conv6", ConvBlock(self.seqy3_2[-1].out_channels, 256 // kd))
        self.seqy3_2.add_module("conv7", ConvBlock(self.seqy3_2[-1].out_channels, self.yolo_layer_input_size))

        self.yolo1, self.yolo2, self.yolo3 = self._create_yolo_layers()

    def _forward_encoder(self, x):
        x = self.conv1(x)

        downs = []
        for module in self.down:
            x, sub = module(x)
            downs.append(sub)

        x = self.seq(x)

        branch1 = self.seq_y1(x)

        x = self.seqy2_1(x)
        x = self.seqy2_2([x, downs[3]])
        branch2 = self.seqy2_3(x)

        x = self.seqy3_1(x)
        branch3 = self.seqy3_2([x, downs[2]])

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
        raise NotImplementedError()

    def save_darknet_weights(self, path, warnings=True):
        raise NotImplementedError()


if __name__ == "__main__":
    from torchsummary import summary

    device = "cuda"
    model = YOLOv3(
        n_class=80,
        onnx=False,
        in_shape=(1, 3, 320, 320),
        anchors=[[(10, 13), (16, 30), (33, 23)], [(30, 61), (62, 45), (59, 119)], [(116, 90), (156, 198), (373, 326)]],
    ).to(device)
    dummy = torch.rand((1, 3, 320, 320)).to(device)
    summary(model, input_size=(3, 416, 416))
