from torch import nn

from pytorch_yolo.models.encoders.darknet import DarknetTinyV3
from pytorch_yolo.models.decoders.yolo import TinyV3Decoder


class TinyV3(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=80,
        anchors=(((10, 14), (23, 27), (37, 58)), ((81, 82), (135, 169), (344, 319))),
        activation=None,
    ):
        super().__init__()

        self.encoder = DarknetTinyV3(in_channels)
        self.decoder = TinyV3Decoder(
            self.encoder.out_channels, num_classes=num_classes, anchors=anchors, activation=activation
        )

    def forward(self, x):
        image_shape = x.shape[-2:]

        x = self.encoder(x)
        x = self.decoder(x, image_shape)
        return x


if __name__ == "__main__":
    from torchsummary import summary

    model = TinyV3().eval().cuda()
    summary(model, (3, 416, 416))
