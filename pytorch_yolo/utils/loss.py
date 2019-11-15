import torch

from torch import nn
from torch import Tensor
from torch.nn.modules.loss import _Loss


def wh_iou(box1, box2):
    box2 = box2.t()

    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]

    # Intersection area
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    # Union Area
    union_area = (w1 * h1 + 1e-6) + w2 * h2 - inter_area

    return inter_area / union_area


class YOLOLoss(_Loss):
    def __init__(self, yolo_layers, binary=False, xy=1, wh=1, cls=1, conf=1, iou_thres=0.1, cls_weights=None,
                 cls_loss=nn.BCEWithLogitsLoss()):
        super().__init__()
        self.xy = xy
        self.wh = wh
        self.cls = cls
        self.conf = conf

        self.yolo_layers = yolo_layers

        self.iou_thres = iou_thres
        self.binary = binary
        self.cls_weights = cls_weights
        self.cls_loss = cls_loss

        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.obj_scale = 1
        self.noobj_scale = 100

    def forward(self, y_pred, y_true):
        total_loss = 0

        for pred, yolo in zip(y_pred, self.yolo_layers):
            anchors = torch.tensor(yolo.scaled_anchors, dtype=torch.float32, device=pred.device)
            obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = self.build_targets(pred, y_true, anchors)

            x = pred[..., 0]
            y = pred[..., 1]
            h = pred[..., 2]
            w = pred[..., 3]
            conf = pred[..., 4]
            cls = pred[..., 5:]

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse(x[obj_mask], tx[obj_mask]) * self.xy
            loss_y = self.mse(y[obj_mask], ty[obj_mask]) * self.xy
            loss_w = self.mse(w[obj_mask], tw[obj_mask]) * self.wh
            loss_h = self.mse(h[obj_mask], th[obj_mask]) * self.wh

            loss_conf_obj = self.bce(conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce(conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            # loss_conf = loss_conf_obj
            loss_conf *= self.conf

            loss_cls = self.cls_loss(cls[obj_mask], tcls[obj_mask]) * self.cls

            total_loss += loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        return total_loss

    def build_targets(self, logits, targets, anchors):
        shape = logits.shape[:-1]
        num_classes = logits.shape[-1] - 5
        device = logits.device

        nG = torch.tensor(logits.shape[-3:-1], device=device, dtype=torch.float32)

        # Output tensors
        obj_mask = torch.zeros(shape, device=device, dtype=torch.bool)
        noobj_mask = torch.ones(shape, device=device, dtype=torch.bool)
        tx = torch.zeros(shape, device=device, dtype=torch.float32)
        ty = torch.zeros(shape, device=device, dtype=torch.float32)
        tw = torch.zeros(shape, device=device, dtype=torch.float32)
        th = torch.zeros(shape, device=device, dtype=torch.float32)
        tcls = torch.zeros(shape + (num_classes, ), device=device, dtype=torch.float32)

        # Convert to position relative to box
        target_boxes = targets[:, 2:6]
        gxy = target_boxes[:, :2] * nG
        gwh = target_boxes[:, 2:] * nG

        # Get anchors with best iou
        ious = torch.stack([wh_iou(anchor, gwh) for anchor in anchors])
        best_ious, best_n = ious.max(0)

        # Separate target values
        b, target_labels = targets[:, :2].long().t()
        gx, gy = gxy.t()
        gw, gh = gwh.t()
        gi, gj = gxy.long().t()

        # Set masks
        obj_mask[b, best_n, gj, gi] = 1
        noobj_mask[b, best_n, gj, gi] = 0

        # Set noobj mask to zero where iou exceeds ignore threshold
        for i, anchor_ious in enumerate(ious.t()):
            noobj_mask[b[i], anchor_ious > self.iou_thres, gj[i], gi[i]] = 0

        # Coordinates
        tx[b, best_n, gj, gi] = gx - gx.floor()
        ty[b, best_n, gj, gi] = gy - gy.floor()

        # Width and height
        tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
        th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)

        # One-hot encoding of label
        tcls[b, best_n, gj, gi, target_labels] = 1

        tconf = obj_mask.float()
        return obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
