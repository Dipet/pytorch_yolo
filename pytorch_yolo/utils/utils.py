import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import tempfile
from pycocotools.cocoeval import COCOeval

try:
    import torch_utils
except:
    from . import torch_utils

import json
from tqdm import tqdm


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)


def xyxy2xywh(x):
    """Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h].

    Args:
        x: annotation in xyxy format.

    Returns: annotation in xyhw format.

    """
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    """Convert bounding box format from [x, y, h, w] to [x1, y1, x2, y2].

        Args:
            x: annotation in xyhw format.

        Returns: annotation in xyxy format.

    """
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def bbox_iou(box1, box2, x1y1x2y2=True):
    """Returns the IoU of box1 to box2. box1 is 4, box2 is nx4

    Args:
        box1:
        box2:
        x1y1x2y2:

    Returns:

    """
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    return inter_area / union_area  # iou


def wh_iou(box1, box2):
    """Returns the IoU of wh1 to wh2. wh1 is 2, wh2 is nx2

    Args:
        box1:
        box2:

    Returns:

    """
    box2 = box2.t()

    # w, h = box1
    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]

    # Intersection area
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    # Union Area
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area  # iou


def compute_loss(p, targets, model, class_weight=None):
    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
    lxy, lwh, lcls, lconf = ft([0]), ft([0]), ft([0]), ft([0])
    txy, twh, tcls, indices = build_targets(model, targets)

    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss(weight=class_weight)
    bce_loss = nn.BCEWithLogitsLoss()

    # Compute losses
    h = model.hyper_params  # hyperparameters
    bs = p[0].shape[0]  # batch size
    k = bs  # loss gain
    for i, pi0 in enumerate(p):  # layer i predictions, i
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tconf = torch.zeros_like(pi0[..., 0])  # conf

        # Compute losses
        if len(b):  # number of targets
            pi = pi0[b, a, gj, gi]  # predictions closest to anchors
            tconf[b, a, gj, gi] = 1  # conf

            lxy += (k * h["xy_loss"]) * mse_loss(torch.sigmoid(pi[..., 0:2]), txy[i])
            lwh += (k * h["wh_loss"]) * mse_loss(pi[..., 2:4], twh[i])

            if model.n_class > 1:
                lcls += (k * h["cls_loss"]) * ce_loss(pi[..., 5:], tcls[i])
            else:
                lcls += (k * h["cls_loss"]) * bce_loss(pi[..., 5], tcls[i].float())

        lconf += (k * h["conf_loss"]) * bce_loss(pi0[..., 4], tconf)
    loss = lxy + lwh + lconf + lcls

    return loss, torch.cat((lxy, lwh, lconf, lcls, loss)).detach()


def build_targets(model, targets):
    # targets = [image, class, x, y, w, h]
    iou_thres = model.hyper_params["iou_thresh"]

    nt = len(targets)
    txy, twh, tcls, indices = [], [], [], []
    for layer in model.yolo_layers:
        # iou of targets-anchors
        t, a = targets, []
        gwh = targets[:, 4:6] * layer.n_grids
        if nt:
            iou = [wh_iou(x, gwh) for x in layer.anchor_vec]
            iou, a = torch.stack(iou, 0).max(0)  # best iou and anchor

            # reject below threshold ious (OPTIONAL, increases P, lowers R)
            reject = True
            if reject:
                j = iou > iou_thres
                t, a, gwh = targets[j], a[j], gwh[j]

        # Indices
        b, c = t[:, :2].long().t()  # target image, class
        gxy = t[:, 2:4] * layer.n_grids  # grid x, y
        gi, gj = gxy.long().t()  # grid x, y indices
        indices.append((b, a, gj, gi))

        # XY coordinates
        txy.append(gxy - gxy.floor())

        # Width and height
        twh.append(torch.log(gwh / layer.anchor_vec[a]))  # wh yolo method

        # Class
        tcls.append(c)
        if c.shape[0]:
            assert c.max() <= layer.n_classes, "Target classes exceed model classes"

    return txy, twh, tcls, indices


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.5):
    """Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.

    Returns: detections with shape
             (x1, y1, x2, y2, object_conf, class_conf, class)
    """
    min_wh = 2  # (pixels) minimum box width and height

    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):
        # Multiply conf by class conf to get combined confidence
        class_conf, class_pred = pred[:, 5:].max(1)
        pred[:, 4] *= class_conf

        # Select only suitable predictions
        i = pred[:, 4] > conf_thres
        i &= (pred[:, 2:4] > min_wh).all(1)
        i &= torch.isfinite(pred).all(1)

        pred = pred[i]

        # If none are remaining => process next image
        if len(pred) == 0:
            continue

        # Select predicted classes
        class_conf = class_conf[i]
        class_pred = class_pred[i].unsqueeze(1).float()

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])

        # Detections ordered as (x1y1x2y2, obj_conf, class_conf, class_pred)
        pred = torch.cat((pred[:, :5], class_conf.unsqueeze(1), class_pred), 1)

        # Get detections sorted by decreasing confidence scores
        pred = pred[(-pred[:, 4]).argsort()]

        det_max = []
        nms_style = "MERGE"  # 'OR' (default), 'AND', 'MERGE' (experimental)
        for c in pred[:, -1].unique():
            dc = pred[pred[:, -1] == c]  # select class c
            n = len(dc)
            if n == 1:
                det_max.append(dc)  # No NMS required if only 1 prediction
                continue
            elif n > 100:
                # limit to first 100 boxes:
                # https://github.com/ultralytics/yolov3/issues/117
                dc = dc[:100]

            # Non-maximum suppression
            if nms_style == "OR":  # default
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold
            elif nms_style == "AND":  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold
            elif nms_style == "MERGE":  # weighted mixture box
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    i = bbox_iou(dc[0], dc) > nms_thres  # iou with other boxes
                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]
            # soft-NMS https://arxiv.org/abs/1704.04503
            elif nms_style == "SOFT":
                sigma = 0.5  # soft-nms sigma parameter
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    det_max.append(dc[:1])
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:]
                    # decay confidences
                    dc[:, 4] *= torch.exp(-iou ** 2 / sigma)

        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate
            output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort

    return output


def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords1 (xyxy) from img1_shape to img0_shape
    gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
    coords[:, [0, 2]] -= (img1_shape[1] - img0_shape[1] * gain) / 2  # x padding
    coords[:, [1, 3]] -= (img1_shape[0] - img0_shape[0] * gain) / 2  # y padding
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords


def _dict_from_results(data, targets, imgs_path, orig_shapes, cur_shape):
    for i, pred in enumerate(targets):
        if pred is None:
            continue

        img_path = imgs_path[i]
        orig_shape = orig_shapes[i]
        pred[:, :4] = scale_coords(cur_shape, pred[:, :4], orig_shape).round()
        for x1, y1, x2, y2, conf, cls_conf, cls in pred.detach().cpu().numpy():
            sub_data = {
                "type": int(cls),
                "score": float(conf),
                "left": int(x1),
                "top": int(y1),
                "right": int(x2),
                "bottom": int(y2),
            }

            if img_path in data:
                data[img_path].append(sub_data)
            else:
                data[img_path] = [sub_data]

    return data


def bench_results(results_path, cocoGt):
    cocoDt = cocoGt.loadRes(results_path)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    iou = 0
    for _, i in cocoEval.ious.items():
        if not isinstance(i, np.ndarray):
            continue
        i = i[i >= 0.3]

        if len(i) < 1:
            continue

        iou += i.mean()
    mean_iou = iou / len(cocoEval.ious)
    print(f"Mean IOU: {mean_iou:.2f}")

    metrics = ["AP", "AP50", "AP75", "APS", "APM", "APL", "AR1", "AR10", "AR100", "ARS", "ARM", "ARL", "IOU"]

    return dict(zip(metrics, list(cocoEval.stats) + [mean_iou]))


def test_model(model: torch.nn.Module, dataset, batch_size, num_workers, device, conf_thresh=0.1, nms_thresh=0.1):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    state = model.training
    model.eval()

    df_predicted = {}

    for imgs, targets, imgs_path, shapes in tqdm(dataloader, total=len(dataloader), desc="Validation"):
        imgs = imgs.to(device)
        with torch.no_grad():
            p, _ = model(imgs)

        det = non_max_suppression(p, conf_thresh, nms_thresh)
        df_predicted = _dict_from_results(df_predicted, det, imgs_path, shapes, imgs.shape[-2:])

    results = tempfile.NamedTemporaryFile("w+")

    json_results = coco_helper.results_from_dict(df_predicted, dataset.coco.dataset)
    json.dump(json_results, results)
    results.flush()

    metrics = bench_results(results.name, dataset.coco)

    if state:
        model.train()

    return metrics
