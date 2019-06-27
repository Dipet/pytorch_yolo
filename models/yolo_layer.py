import torch
from torch import nn
from torch.nn import functional as F


class Upsample(nn.Module):
    def __init__(self, scale_factor=1, mode='nearest'):
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
    def __init__(self, anchors, nc, all_anchors,
                 onnx=False, in_tensor=None, img_size=None):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)
        self.n_anchors = len(anchors)
        self.anchor_wh = 0
        self.n_classes = nc
        self.n_grids = 0
        self.n_x_grids = 0
        self.n_y_grids = 0

        self.img_size = 0
        self.stride = 0
        self.grid_xy = 0
        self.anchor_vec = 0

        self.onnx = onnx

        self.all_anchors = all_anchors

        if onnx and (in_tensor is None or img_size is None):
            raise ValueError('With onnx flag need in_tensor and img_size')
        elif onnx:
            bs, ch, ny, nx = in_tensor.shape

            self.img_size = img_size
            self.n_x_grids = nx
            self.n_y_grids = ny
            self.create_grids(in_tensor.device)

    def forward(self, p, img_size):
        bs, ny, nx = p.shape[0], p.shape[-2], p.shape[-1]
        if not self.onnx and (self.n_x_grids, self.n_y_grids) != (nx, ny):
            self.img_size = img_size
            self.n_x_grids = nx
            self.n_y_grids = ny
            self.create_grids(p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)
        # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.n_anchors, self.n_classes + 5,
                   self.n_y_grids, self.n_x_grids)\
            .permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p
        elif self.onnx:
            # Constants CAN NOT BE BROADCAST, ensure correct shape!
            ngu = self.n_grids.repeat((bs, self.n_anchors * self.n_x_grids * self.n_y_grids, 1)).to(p.device)
            grid_xy = self.grid_xy.repeat((bs, self.n_anchors, 1, 1, 1)).view((bs, -1, 2)).to(p.device)
            anchor_wh = self.anchor_wh.repeat((bs, 1, self.n_x_grids, self.n_y_grids, 1)).view((bs, -1, 2)).to(p.device) / ngu

            p = p.view(bs, -1, 5 + self.n_classes)
            xy = torch.sigmoid(p[..., 0:2]) + grid_xy  # x, y
            wh = torch.exp(p[..., 2:4]) * anchor_wh  # width, height
            p_conf = torch.sigmoid(p[..., 4:5])  # Conf
            p_cls = p[..., 5:]

            p_cls = torch.exp(p_cls).permute((2, 1, 0))
            p_cls = p_cls / p_cls.sum(0).unsqueeze(0) * p_conf.permute((2, 1, 0))  # F.softmax() equivalent
            p_cls = p_cls.permute(2, 1, 0)
            return torch.transpose(torch.cat((xy / ngu, wh, p_conf, p_cls), 2).squeeze(), -2, -1)

        io = p.clone()  # inference output
        io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + self.grid_xy  # xy
        io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
        io[..., 4:] = torch.sigmoid(io[..., 4:])  # p_conf, p_cls
        io[..., :4] *= self.stride
        if self.n_classes == 1:
            io[..., 5] = 1  # single-class model https://github.com/ultralytics/yolov3/issues/235

        # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
        return io.view(bs, -1, 5 + self.n_classes), p

    def create_grids(self, device='cpu'):
        self.stride = self.img_size / max(self.n_x_grids, self.n_y_grids)

        # build xy offsets
        yv, xv = torch.meshgrid([torch.arange(self.n_y_grids), torch.arange(self.n_x_grids)])
        self.grid_xy = torch.stack((xv, yv), 2).to(device).float().view((1, 1, self.n_y_grids, self.n_x_grids, 2))

        # build wh gains
        self.anchor_vec = self.anchors.to(device) / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.n_anchors, 1, 1, 2).to(device)
        self.n_grids = torch.Tensor((self.n_x_grids, self.n_y_grids)).to(device)