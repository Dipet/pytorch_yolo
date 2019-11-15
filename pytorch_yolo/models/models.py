import torch

from torch import nn

from pytorch_yolo.models.encoders.darknet import DarknetV3, DarknetTinyV3
from pytorch_yolo.models.decoders.yolo import YoloV3Decoder, TinyV3Decoder


class YoloBaseModel(nn.Module):
    def forward(self, x, predict=False):
        batch_size = len(x)
        image_shape = x.shape[-2:]

        x = self.encoder(x)
        x = self.decoder(x, image_shape, predict=predict)

        if predict:
            return torch.cat(x, dim=1)

        return x

    def predict(self, x):
        return self(x, predict=True)


class YoloV3(YoloBaseModel):
    def __init__(
            self,
            in_channels=3,
            num_classes=80,
            anchors=(((10, 13), (16, 30), (33, 23)), ((30, 61), (62, 45), (59, 119)), ((116, 90), (156, 198), (373, 326))),
            activation=None,
    ):
        super().__init__()

        self.encoder = DarknetV3(in_channels)
        self.decoder = YoloV3Decoder(
            self.encoder.out_channels, num_classes=num_classes, anchors=anchors, activation=activation
        )


class TinyV3(YoloBaseModel):
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
