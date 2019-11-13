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
    def __init__(self, anchosrs, binary=False, xy=1, wh=1, cls=1, conf=1, iou_thres=0.1, cls_weights=None,
                 cls_loss=nn.BCEWithLogitsLoss()):
        super().__init__()
        self.xy = xy
        self.wh = wh
        self.cls = cls
        self.conf = conf
        self.anchors = torch.tensor(anchosrs, dtype=torch.float32)
        self.iou_thres = iou_thres
        self.binary = binary
        self.cls_weights = cls_weights
        self.cls_loss = cls_loss

        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.obj_scale = 1
        self.noobj_scale = 100

    def old(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        lxy, lwh, lcls, lconf = torch.zeros(4).to(y_pred[0].device)
        txy, twh, tcls, indices = self.build_targets(y_pred, y_true)

        mse_loss = nn.MSELoss()
        ce_loss = nn.CrossEntropyLoss(weight=self.cls_weights)
        bce_loss = nn.BCEWithLogitsLoss()

        # Compute losses
        bs = y_pred[0].shape[0]
        k = bs  # loss gain
        for i, pi0 in enumerate(y_pred):
            b, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tconf = torch.zeros_like(pi0[..., 0])  # conf

            # Compute losses
            if len(b):  # number of targets
                pi = pi0[b, gj, gi]  # predictions closest to anchors
                tconf[b, gj, gi] = 1  # conf

                lxy += k * self.xy * mse_loss(torch.sigmoid(pi[..., 0:2]), txy[i])
                lwh += k * self.wh * mse_loss(pi[..., 2:4], twh[i])

                if self.binary:
                    lcls += k * self.cls * bce_loss(pi[..., 5], tcls[i].float())
                else:
                    lcls += k * self.cls * ce_loss(pi[..., 5:], tcls[i])


            lconf += k * self.conf * bce_loss(pi0[..., 4], tconf)
        loss = lxy + lwh + lconf + lcls

        return loss

    def new(self, y_pred: Tensor, y_true: Tensor):
        total_loss = 0

        for pred, anchors in zip(y_pred, self.anchors):
            obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = self.__build_targets(pred, y_true, anchors)

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

    def forward(self, y_pred, y_true):
        if self.anchors.device != y_true.device:
            self.anchors = self.anchors.to(y_true.device)

        return self.new(y_pred, y_true)

    def __build_targets(self, logits, targets, anchors):
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

    def build_targets(self, logits, targets):
        if self.anchors.device != targets.device:
            self.anchors = self.anchors.to(targets.device)

        txy, twh, tcls, indices = [], [], [], []

        for anchors, logit in zip(self.anchors, logits):
            n_grids = torch.tensor(logit.shape[-3:-1], device=targets.device, dtype=torch.float32)

            # iou of targets-anchors
            t = targets
            gxy = t[:, 2:4] * n_grids
            gwh = t[:, 4:6] * n_grids

            iou = torch.stack([wh_iou(x, gwh) for x in anchors], 0)
            iou, best_n = iou.max(0)

            j = iou > self.iou_thres
            t, gxy, gwh = t[j], gxy[j], gwh[j]

            # Indices
            b, c = t[:, :2].long().t()  # target image, class
            gi, gj = gxy.long().t()  # grid x, y indices
            indices.append((b, gj, gi))

            # # XY coordinates
            txy.append(gxy - gxy.floor())

            # # Width and height
            twh.append(gwh)  # wh yolo method

            # # Class
            tcls.append(c)

        return txy, twh, tcls, indices