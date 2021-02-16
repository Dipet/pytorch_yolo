import torch
import numpy as np

from torch import nn

from pytorch_yolo.models.encoders.darknet import DarknetV3, DarknetTinyV3
from pytorch_yolo.models.decoders.yolo import YoloV3Decoder, TinyV3Decoder


class YoloBaseModel(nn.Module):
    def forward(self, x, predict=False):
        image_shape = x.shape[-2:]

        x = self.encoder(x)
        x = self.decoder(x, image_shape, predict=predict)

        if predict:
            return torch.cat(x, dim=1)

        return x

    def predict(self, x):
        return self(x, predict=True)

    def load_darknet_weights(self, path):
        with open(path, "rb") as file:
            np.fromfile(file, dtype=np.int32, count=5)  # header
            weights = np.fromfile(file, dtype=np.float32)

        weights = self.encoder.load_darknet_weights(weights)
        self.decoder.load_darknet_weights(weights)


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


if __name__ == '__main__':
    import sys
    sys.path = ['/home/druzhinin/HDD/GitRepos/PyTorch-YOLOv3'] + sys.path
    from torchsummary import summary

    from models import Darknet

    weights = '/home/druzhinin/HDD/GitRepos/yolov3/weights/yolov3.weights'

    model = YoloV3()
    model.load_darknet_weights(weights)

    model.eval()
    summary(model, (3, 320, 320), device='cpu')
