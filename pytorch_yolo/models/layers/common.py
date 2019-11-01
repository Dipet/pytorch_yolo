import torch
import torch.nn.functional as F
from torch import nn


class Upsample(nn.Module):
    def __init__(self, scale_factor=1, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class Concat(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, args):
        return torch.cat(args, self.dim)


class Add(nn.Module):
    def forward(self, x, y):
        return x + y
