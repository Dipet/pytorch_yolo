import os
import numpy as np
import torch

from torch import nn

from collections import OrderedDict

try:
    from ..utils.torch_utils import fuse_conv_and_bn
    from .yolo_layer import YOLOLayer
except:
    from utils.torch_utils import fuse_conv_and_bn
    from models.yolo_layer import YOLOLayer

from abc import abstractclassmethod


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 size=3,
                 stride=1,
                 pad=True):
        super().__init__()

        conv_pad = (size - 1) // 2 if pad else 0

        self.sequence = nn.Sequential(OrderedDict((
            ('conv', nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=size,
                               stride=stride,
                               padding=conv_pad,
                               bias=False)),
            ('batch_norm', nn.BatchNorm2d(out_channels)),
            ('activation', nn.LeakyReLU(0.1, inplace=True)),
        )))

        self.out_channels = out_channels

    def forward(self, x):
        return self.sequence(x)

    def fuse(self):
        fused_sequence = []
        for i, layer in enumerate(self.sequence):
            if isinstance(layer, nn.BatchNorm2d):
                # fuse this bn layer with the previous conv2d layer
                conv = self.sequence[i - 1]
                layer = fuse_conv_and_bn(conv, layer)
                fused_sequence = fused_sequence[:-1] + [layer]
                continue

            fused_sequence.append(layer)
        self.sequence = nn.Sequential(*fused_sequence)


class MaxPool(nn.MaxPool2d):
    def __init__(self, size, stride):
        if size == 2 and stride == 1:
            super().__init__(size, stride, padding=1, dilation=2)
        else:
            pad = (size - 1) // 2
            super().__init__(size, stride, padding=pad)


class ConvPoolBlock(ConvBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_size=3,
                 conv_stride=1,
                 conv_pad=True,
                 pool_size=2,
                 pool_stride=2):
        super().__init__(in_channels, out_channels,
                         conv_size, conv_stride, conv_pad)
        self.sequence.add_module('max_pool', MaxPool(pool_size, pool_stride))


class YOLOBase(nn.Module):
    def __init__(self,
                 in_channels=3,
                 n_class=80,
                 kernels_divider=1,
                 anchors=(((10.0, 14.0), (23.0, 27.0), (37.0, 58.0)),
                          ((81.0, 82.0), (135.0, 169.0), (344.0, 319.0))),
                 onnx=False,
                 in_shape=None,
                 hyper_params=None):
        super().__init__()

        self.header_info = np.zeros(5, dtype=np.int32)
        self.seen = self.header_info[3]
        self.hyper_params = hyper_params
        self.n_class = n_class
        self.onnx = onnx
        self.in_channels = in_channels
        self.anchors = anchors
        self.kernels_divider = kernels_divider
        self.in_shape = in_shape

        self.yolo_layer_input_size = 15 + 3 * n_class

        if onnx and self.in_shape is None:
            raise ValueError("With onnx flag need in_shape value")

        self.encoder = None

    def _create_yolo_layers(self, device='cpu'):
        layers = []

        dummies = [None] * len(self.anchors)
        if self.onnx:
            dummy = torch.rand(self.in_shape).to(device)
            dummies = self._forward_encoder(dummy)

        for anchor, dummy in zip(self.anchors, dummies):
            if self.onnx:
                layers.append(YOLOLayer(anchor, self.n_class,
                                        self.anchors,
                                        self.onnx,
                                        dummy,
                                        max(self.in_shape[-2:])))
            else:
                layers.append(YOLOLayer(anchor, self.n_class,
                                        self.anchors, self.onnx))

        return layers

    def forward(self, _):
        raise NotImplementedError

    def load_darknet_weights(self, weights_path, warnings=True):
        raise NotImplementedError

    @abstractclassmethod
    def _forward_encoder(self, x):
        raise NotImplementedError

    @property
    def yolo_layers(self):
        raise NotImplementedError

    def _load_darknet_weights(self, weights_path, layers, warnings=True):
        """Parses and loads the weights stored in 'weights'

        Args:
            weights_path: path ot *.weights file
            type: supported types ['tiny', 'full']

        """
        _, weights_file = os.path.split(weights_path)

        # Try to download weights if not available locally
        if not os.path.isfile(weights_path):
            try:
                os.system('wget https://pjreddie.com/media/files/'
                          + weights_file + ' -O ' + weights_path)
            except IOError:
                print(f'{weights_path} not found.')
                print('Try https://drive.google.com/drive/folders/1uxgUBemJVw9wZsdpboYbzUN4bcRhsuAI')

        # Open the weights file
        with open(weights_path, 'rb') as f:
            # First five are header values
            header = np.fromfile(f, dtype=np.int32,  count=5)

            # Needed to write header when saving weights
            self.header_info = header
            # number of images seen during training
            self.seen = header[3]
            # The rest are weights
            weights_path = np.fromfile(f, dtype=np.float32)

        ptr = 0
        for layer in layers:
            if not isinstance(layer, (nn.Conv2d, ConvBlock, ConvPoolBlock)):
                if warnings:
                    print('Skip unsupported layer:', str(layer))
                continue

            if isinstance(layer, nn.Conv2d):
                # Load conv. bias
                num_b = layer.bias.numel()
                conv_b = torch.from_numpy(weights_path[ptr:ptr + num_b])
                conv_b = conv_b.view_as(layer.bias)
                layer.bias.data.copy_(conv_b)
                ptr += num_b
            else:
                # Load BN bias, weights, running mean and running variance
                bn_layer = layer[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights_path[ptr:ptr + num_b])
                bn_b = bn_b.view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights_path[ptr:ptr + num_b])
                bn_w = bn_w.view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights_path[ptr:ptr + num_b])
                bn_rm = bn_rm.view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights_path[ptr:ptr + num_b])
                bn_rv = bn_rv.view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b

            # Load conv. weights
            conv_layer = layer[0]
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights_path[ptr:ptr + num_w])
            conv_w = conv_w.view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w

    def save_darknet_weights(self, path, warnings=True):
        raise NotImplementedError

    def _save_weights(self, path, layers, warnings=True):
        """Converts a PyTorch model to Darket format (*.pt to *.weights)

        Args:
            path: save path.
            cutoff: layers cutoff.

        """
        with open(path, 'wb') as f:
            self.header_info[3] = self.seen
            self.header_info.tofile(f)

            # Iterate through layers
            for layer in layers:
                if not isinstance(layer, (nn.Conv2d, ConvBlock, ConvPoolBlock)):
                    if warnings:
                        print('Skip unsupported layer:', str(layer))
                    continue

                # If batch norm, load bn first
                if isinstance(layer, (ConvBlock, ConvPoolBlock)):
                    bn_layer = layer[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)

                    layer[0].weight.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    layer.bias.data.cpu().numpy().tofile(f)
                    # Load conv weights
                    layer.weight.data.cpu().numpy().tofile(f)
