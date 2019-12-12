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
    def __init__(self, yolo_layers, bbox_weight=1, cls_weight=10, obj_weight=1, no_obj_weight=10, iou_thres=0.1,
                 cls_loss=nn.BCEWithLogitsLoss(), extended=False,
                 best_anchors=False, best_all_anchors=False):
        super().__init__()
        self.bbox_weight = bbox_weight
        self.cls_weight = cls_weight

        self.obj_weight = obj_weight
        self.no_obj_weight = no_obj_weight

        self.yolo_layers = yolo_layers

        self.iou_thres = iou_thres
        self.cls_loss = cls_loss

        self.best_anchors = best_anchors
        self.best_all_anchors = best_all_anchors

        self.extended = extended

        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    @staticmethod
    def bbox_iou(box1, box2):
        eps = torch.finfo(box1.dtype).eps
        max_val = torch.finfo(box1.dtype).max

        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
        box2 = box2.t()

        # Get the coordinates of bounding boxes
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        # Intersection area
        inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                     (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        union_area = w1 * h1 + w2 * h2 - inter_area
        union_area = torch.clamp(union_area, eps, max_val)

        iou = inter_area / union_area  # iou

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

        # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
        c_area = torch.clamp(cw * ch, eps, max_val)  # convex area
        iou = iou - 1 + union_area / c_area  # GIoU

        return iou

    def forward(self, y_pred, y_true):
        loss_dict = {}
        loss = 0
        dtype = y_pred[0].dtype
        device = y_pred[0].device
        all_anchors = [torch.tensor(yolo.scaled_anchors, dtype=dtype, device=device) for yolo in self.yolo_layers]
        all_num_grids = [torch.tensor(logits.shape[-3:-1], device=device, dtype=dtype) for logits in y_pred]

        # Compute losses
        for pred, num_grids, anchors in zip(y_pred, all_num_grids, all_anchors):
            target_cls, target_bbox, indices, targets_anchors = self.build_targets(num_grids, all_num_grids, y_true, anchors, all_anchors, dtype)

            image, anchor, grid_y, grid_x = indices
            target_obj = torch.zeros_like(pred[..., 0])

            # Compute losses
            num_targets = len(image)
            if num_targets:
                pred_subset = pred[image, anchor, grid_y, grid_x]  # prediction subset corresponding to targets
                target_obj[image, anchor, grid_y, grid_x] = 1.0  # obj

                # GIoU
                pxy = pred_subset[:, 0:2]
                pbox = torch.cat((pxy, pred_subset[:, 2:4].clamp(max=1E4) * targets_anchors), 1)  # predicted box
                giou = self.bbox_iou(pbox.t(), target_bbox)  # giou computation
                loss_bbox = (1.0 - giou).mean()  # giou loss

                t = torch.zeros_like(pred_subset[:, 5:])  # targets
                t[range(num_targets), target_cls] = 1.0

                loss_cls = self.bce(pred_subset[:, 5:], t)
            else:
                loss_bbox = 0
                loss_cls = 0

            obj_mask = torch.zeros_like(target_obj, dtype=torch.bool)
            obj_mask[image, anchor, grid_y, grid_x] = True

            no_obj_mask = ~obj_mask

            if obj_mask.any():
                loss_has_obj = self.bce(pred[..., 4][obj_mask], target_obj[obj_mask]) * self.obj_weight
            else:
                loss_has_obj = 0
            if no_obj_mask.any():
                loss_no_obj = self.bce(pred[..., 4][no_obj_mask], target_obj[no_obj_mask]) * self.no_obj_weight
            else:
                loss_no_obj = 0
            loss_obj =  loss_has_obj + loss_no_obj

            loss_bbox *= self.bbox_weight
            loss_cls *= self.cls_weight

            loss += loss_bbox + loss_obj + loss_cls

            ngrids = pred.shape[-3:-1]
            if ngrids not in loss_dict:
                loss_dict[ngrids] = {
                    'loss': 0,
                    'bbox': 0,
                    'obj': 0,
                    'has_obj': 0,
                    'no_obj': 0,
                    'cls': 0
                }

            loss_dict[ngrids]['obj'] += float(loss_obj)
            loss_dict[ngrids]['has_obj'] += float(loss_has_obj)
            loss_dict[ngrids]['no_obj'] += float(loss_no_obj)
            loss_dict[ngrids]['bbox'] += float(loss_bbox)
            loss_dict[ngrids]['cls'] += float(loss_cls)
            loss_dict[ngrids]['loss'] += float(loss_obj) + float(loss_bbox) + float(loss_cls)

            if torch.isnan(loss).any() or not torch.isfinite(loss).any():
                raise ValueError("Nan in loss")

        if self.extended:
            return loss, loss_dict

        return loss

    def build_targets(self, num_grids, all_num_grids, targets, anchors, all_anchors, dtype):
        num_targets = len(targets)
        targets = targets.to(dtype)
        anchors = anchors.to(dtype)

        # iou of targets-anchors
        t, targets_anchors = targets, []
        gwh = t[:, 4:6] * num_grids
        if num_targets:
            iou = torch.stack([wh_iou(x, gwh) for x in anchors], 0)
            other_iou = []

            if self.best_all_anchors:
                other_iou = [torch.stack([wh_iou(x, t[:, 4:6] * ng) for x in a], 0) for a, ng in zip(all_anchors, all_num_grids) if (a != anchors).any()]
                other_iou = [i.max(0).values for i in other_iou]

                iou, targets_anchors = iou.max(0)
            elif self.best_anchors:
                iou, targets_anchors = iou.max(0)
            else:
                na = len(anchors)
                targets_anchors = torch.arange(na).view((-1, 1)).repeat([1, num_targets]).view(-1)
                t = targets.repeat([na, 1])
                gwh = gwh.repeat([na, 1])
                iou = iou.view(-1)  # use all ious

            # reject anchors below iou_thres (OPTIONAL, increases P, lowers R)
            j = iou > self.iou_thres
            if self.best_all_anchors:
                for i in other_iou:
                    j &= iou >= i
            t, targets_anchors, gwh = t[j], targets_anchors[j], gwh[j]

        # Indices
        b, target_cls = t[:, :2].long().t()  # target image, class
        gxy = t[:, 2:4] * num_grids  # grid x, y
        gi, gj = gxy.long().t()  # grid x, y indices
        indices = (b, targets_anchors, gj, gi)

        # GIoU
        gxy -= gxy.floor()  # xy
        target_bbox = torch.cat((gxy, gwh), 1)  # xywh (grids)

        return target_cls, target_bbox, indices, anchors[targets_anchors]
