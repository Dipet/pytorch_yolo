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
    def __init__(self, yolo_layers, binary=False, xy=1, wh=1, cls=1, obj=1, no_obj=1, iou_thres=0.1, cls_weights=None,
                 cls_loss=nn.BCEWithLogitsLoss()):
        super().__init__()
        self.xy = xy
        self.wh = wh
        self.cls = cls
        self.obj = obj
        self.no_obj = no_obj

        self.yolo_layers = yolo_layers

        self.iou_thres = iou_thres
        self.binary = binary
        self.cls_weights = cls_weights
        self.cls_loss = cls_loss

        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        total_loss = 0

        for pred, yolo in zip(y_pred, self.yolo_layers):
            anchors = torch.tensor(yolo.scaled_anchors, dtype=torch.float32, device=pred.device)
            obj_mask, no_obj_mask, tx, ty, tw, th, tcls, tconf = self.build_targets(pred, y_true, anchors)

            x = pred[..., 0]
            y = pred[..., 1]
            h = pred[..., 2]
            w = pred[..., 3]
            conf = pred[..., 4]
            cls = pred[..., 5:]

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            if obj_mask.any():
                loss_x = self.mse(x[obj_mask], tx[obj_mask]) * self.xy
                loss_y = self.mse(y[obj_mask], ty[obj_mask]) * self.xy
                loss_w = self.mse(w[obj_mask], tw[obj_mask]) * self.wh
                loss_h = self.mse(h[obj_mask], th[obj_mask]) * self.wh
                loss_cls = self.cls_loss(cls[obj_mask], tcls[obj_mask]) * self.cls
                loss_conf_obj = self.bce(conf[obj_mask], tconf[obj_mask])
            else:
                loss_x = 0
                loss_y = 0
                loss_w = 0
                loss_h = 0
                loss_cls = 0
                loss_conf_obj = 0

            if no_obj_mask.any():
                loss_conf_noobj = self.bce(conf[no_obj_mask], tconf[no_obj_mask])
            else:
                loss_conf_noobj = 0

            loss_conf = loss_conf_obj * self.obj + loss_conf_noobj * self.no_obj

            total_loss += loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        return total_loss

    def build_targets(self, logits, targets, anchors):
        shape = logits.shape[:-1]
        num_classes = logits.shape[-1] - 5
        device = logits.device

        nG = torch.tensor(logits.shape[-3:-1], device=device, dtype=torch.float32)

        # Output tensors
        obj_mask = torch.zeros(shape, device=device, dtype=torch.bool)
        no_obj_mask = torch.zeros(shape, device=device, dtype=torch.bool)
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

        # Objects masks
        obj_mask[b, best_n, gj, gi] = 1
        no_obj_mask = ~obj_mask

        for i, anchor_ious in enumerate(ious.t()):
            c = anchor_ious < self.iou_thres
            obj_mask[b[i], c, gj[i], gi[i]] = 0
            no_obj_mask[b[i], c, gj[i], gi[i]] = 1
            no_obj_mask[b[i], ~c, gj[i], gi[i]] = 0


        # Coordinates
        tx[b, best_n, gj, gi] = gx - gx.floor()
        ty[b, best_n, gj, gi] = gy - gy.floor()

        # Width and height
        tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
        th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)

        # One-hot encoding of label
        tcls[b, best_n, gj, gi, target_labels] = 1

        tconf = obj_mask.float()
        return obj_mask, no_obj_mask, tx, ty, tw, th, tcls, tconf
