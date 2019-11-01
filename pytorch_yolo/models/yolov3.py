from torch import nn

from pytorch_yolo.models.encoders.darknet import DarknetV3
from pytorch_yolo.models.decoders.yolo import YoloV3Decoder


class YoloV3(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=80,
        anchors=(((10, 13), (16, 30), (33, 23)), ((30, 61), (62, 45), (59, 119)), ((116, 90), (156, 198), (373, 326))),
            activation=None
    ):
        super().__init__()

        self.encoder = DarknetV3(in_channels)
        self.decoder = YoloV3Decoder(self.encoder.out_channels, num_classes=num_classes, anchors=anchors, activation=activation)

    def forward(self, x):
        image_shape = x.shape[-2:]

        x = self.encoder(x)
        x = self.decoder(x, image_shape)
        return x


if __name__ == '__main__':
    from torchsummary import summary

    model = YoloV3().eval().cuda()
    summary(model, (3, 416, 416))
