from pytorch_yolo.models.old.model import Model

from pytorch_yolo.models.encoders.darknet import DarknetTinyV3
from pytorch_yolo.models.decoders.yolo import TinyV3Decoder


class TinyV3(Model):
    def __init__(
        self,
        in_channels=3,
        num_classes=80,
        load_weights=True,
        anchors=(((10, 14), (23, 27), (37, 58)), ((81, 82), (135, 169), (344, 319))),
    ):
        super().__init__(in_channels, num_classes, load_weights, anchors)

        self.encoder = DarknetTinyV3(in_channels, load_weights=load_weights)

        encoder_channels = self.encoder.out_channels
        decoder_in_channels = encoder_channels[8], encoder_channels[-1]

        self.decoder = TinyV3Decoder(decoder_in_channels, num_classes, anchors)

    def forward(self, x):
        outputs = self.encoder(x)

        return self.decoder(outputs[8], outputs[-1])


if __name__ == "__main__":
    from torchsummary.torchsummary import summary

    model = TinyV3()
    summary(model, (3, 416, 416), device="cpu")
