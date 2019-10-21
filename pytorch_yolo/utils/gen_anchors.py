"""
Based on https://github.com/experiencor/keras-yolo2/blob/master/gen_anchors.py
"""

import random
import numpy as np

try:
    from .dataset_csv import CSVDatasetInference
except:
    from dataset_csv import CSVDatasetInference

from torch.utils.data import DataLoader


from tqdm import tqdm
from sklearn.cluster import KMeans


def iou(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)  # will become (k,) shape

    return np.array(similarities)


def avg_iou(anns, centroids):
    n, d = anns.shape
    s = 0.0

    for i in range(anns.shape[0]):
        s += max(iou(anns[i], centroids))

    return s / n


def print_anchors(centroids):
    anchors = centroids.copy()

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    r = "anchors: ["
    for i in sorted_indices[:-1]:
        r += "%0.2f,%0.2f, " % (anchors[i, 0], anchors[i, 1])

    # there should not be comma after last anchor, that's why
    r += "%0.2f,%0.2f" % (
        anchors[sorted_indices[-1:], 0],
        anchors[sorted_indices[-1:], 1],
    )
    r += "]"

    print(r)


def main(train_path, num_anchors, img_size):
    dataset = CSVDatasetInference(train_path, img_size=img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=6,
        collate_fn=dataset.collate_fn,
    )

    # run k_mean to find the anchors
    ann_w = []
    ann_h = []
    for image, labels, im0 in tqdm(dataloader):
        for label in labels:
            label = label.astype(float)
            relative_w = label[:, 3] - label[:, 1]
            relative_h = label[:, 4] - label[:, 2]
            ann_h.append(relative_h)
            ann_w.append(relative_w)

    ann_w = np.concatenate(ann_w)
    ann_h = np.concatenate(ann_h)
    annotation_dims = np.column_stack([ann_w, ann_h])
    # centroids = run_kmeans(annotation_dims, num_anchors)
    kmeans = KMeans(num_anchors, verbose=False).fit(annotation_dims)
    centroids = kmeans.cluster_centers_

    # write anchors to file
    print(
        "\naverage IOU for",
        num_anchors,
        "anchors:",
        "%0.2f" % avg_iou(annotation_dims, centroids),
    )
    print(f"Mean distance: {kmeans.inertia_:.2f}")
    print_anchors(centroids)


if __name__ == "__main__":
    main(
        "/home/druzhinin/HDD/Projects/Detection/Datasets/train/1000_and_10_000_nyc_pascal_nuscenes_mini_cars_cbcl_train_kitti_coco_train_detrac.csv",
        6,
        416,
    )
