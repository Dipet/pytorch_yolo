import torch
from torch import nn
from torch.nn import functional as F


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


class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, class_activation=None):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.tensor(anchors)
        self.num_anchors = len(anchors)
        self.anchor_wh = 0
        self.num_classes = num_classes
        self.nx_grids = 0
        self.ny_grids = 0

        self.img_size = 0
        self.stride = 0
        self.grid_xy = 0
        self.anchor_vec = 0
        self.cls_activation = class_activation

    def forward(self, x, img_size):
        batch_size, ny, nx = x.shape[0], x.shape[-2], x.shape[-1]
        if (self.nx_grids, self.ny_grids) != (nx, ny):
            self.create_grids(img_size, nx, ny, x.device)

        # (batch_size, anchors, grid, grid, xywh + classes)
        x = (
            x.view(batch_size, self.num_anchors, self.num_classes + 5, self.ny_grids, self.nx_grids)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
        )

        if self.cls_activation is not None:
            x[..., 0:2] = torch.sigmoid(x[..., 0:2]) + self.grid_xy  # xy
            x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_wh  # wh yolo method
            x[..., :4] *= self.stride

            x[..., 4] = torch.sigmoid(x[..., 4])  # p_conf
            x[..., 5:] = self.cls_activation(x)

        return x

    def create_grids(self, img_size, nx, ny, device="cpu"):
        self.img_size = img_size
        self.nx_grids = nx
        self.ny_grids = ny

        self.stride = self.img_size / max(self.nx_grids, self.ny_grids)

        # build xy offsets
        yv, xv = torch.meshgrid([torch.arange(self.ny_grids), torch.arange(self.nx_grids)])
        self.grid_xy = torch.stack((xv, yv), 2).to(device).float().view((1, 1, self.ny_grids, self.nx_grids, 2))

        # build wh gains
        self.anchor_vec = self.anchors.to(device) / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.num_anchors, 1, 1, 2).to(device)
