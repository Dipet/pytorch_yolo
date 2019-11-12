import torch
from torch import nn


class YoloLayer(nn.Module):
    def __init__(self, anchors, num_classes=80, class_activation=None):
        super(YoloLayer, self).__init__()

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

        self.input_channels: int = self.num_anchors * (self.num_classes + 5)

    def forward(self, x, img_size, predict=False):
        batch_size, ny, nx = x.shape[0], x.shape[-2], x.shape[-1]
        if (self.nx_grids, self.ny_grids) != (nx, ny):
            self.create_grids(img_size, nx, ny, x.device)

        # (batch_size, anchors, grid, grid, xywh + classes)
        x = (
            x.view(batch_size, self.num_anchors, self.num_classes + 5, self.ny_grids, self.nx_grids)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
        )

        x[..., 0:2] = torch.sigmoid(x[..., 0:2])

        if self.cls_activation is not None:
            x[..., 5:] = self.cls_activation(x[..., 5:])

        if predict:
            x[..., :2] += self.grid_xy
            x[..., 2:4] *= self.anchor_wh
            x[..., :4] = torch.exp(x[..., :4]) * self.stride
            x[..., 4] = torch.sigmoid(x[..., 4])

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
