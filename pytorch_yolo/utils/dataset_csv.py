import os
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

import pandas as pd

import torch

from .utils import coco_helper, xyxy2xywh

import json

from pycocotools.coco import COCO


class CSVDataset(Dataset):  # for training/testing
    def __init__(self, path, transform=None):
        df = pd.read_csv(path)
        if df["type"].dtype == np.object:
            map_type = {j: i for i, j in enumerate(sorted(df["type"].unique()))}
            df["type"] = df["type"].replace(map_type)

        if min(df["type"]) > 0:  # Fix class
            df["type"] -= 1
        self.cls_number = len(df["type"].unique())
        self._dataset = {}
        self.class_weight = self.compute_labels_weights(df["type"])

        dirpath, name = os.path.split(path)
        labels_path = os.path.join(dirpath, "labels_" + name)
        if os.path.exists(labels_path):
            self._labels = pd.read_csv(labels_path)["label"].values
        else:
            self._labels = np.array([str(i) for i in sorted(df["type"].unique())])

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Parse csv file"):
            path = row["image_path"]

            label = [row["left"], row["top"], row["right"], row["bottom"], row["type"]]

            if path in self._dataset:
                self._dataset[path].append(label)
            else:
                self._dataset[path] = [label]

        for i in list(self._dataset.keys()):
            self._dataset[i] = np.array(self._dataset[i]).astype(float)

        n = len(self._dataset)
        assert n > 0, "No images found in %s" % path

        self._img_files = list(self._dataset.keys())

        self.transform = transform

    def _get_data(self, index):
        img_path = self._img_files[index]

        img = cv.imread(img_path, cv.IMREAD_COLOR)  # BGR
        assert img is not None, "File Not Found " + img_path
        labels = self._dataset[img_path].copy()

        labels[:, 1] = np.maximum(labels[:, 1], 0)
        labels[:, 2] = np.maximum(labels[:, 2], 0)
        labels[:, 3] = np.minimum(labels[:, 3], img.shape[1] - 1)
        labels[:, 4] = np.minimum(labels[:, 4], img.shape[0] - 1)

        return img, labels

    @staticmethod
    def _convert_img_for_net(img):
        # Normalize 0-1
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255

        # To Pytorch format
        img = np.transpose(img, (2, 0, 1))

        return torch.from_numpy(img)

    @staticmethod
    def _convert_labels_for_net(labels, img):
        # transform [x1, y1, x2, y2, cls] -> [cls, x1, y1, x2, y2]
        labels[:, 1:], labels[:, 0] = labels[:, :4], labels[:, 4].copy()

        # Convert labels format
        h, w = img.shape[:2]
        labels[:, 1:] = xyxy2xywh(labels[:, 1:])
        labels[:, (1, 3)] = labels[:, (1, 3)] / w
        labels[:, (2, 4)] = labels[:, (2, 4)] / h

        # Add image index axis
        result_labels = np.zeros((len(labels), 6), dtype=np.float32)
        result_labels[:, 1:] = labels

        return torch.from_numpy(result_labels)

    def use_transform(self, img, labels=None):
        if labels is not None:
            data = {"image": img, "bboxes": labels}
        else:
            data = {"image": img}

        if self.transform:
            data = self.transform(**data)

        if labels is not None:
            labels = np.array(data["bboxes"]) if len(data["bboxes"]) else np.zeros((0,) + labels.shape[1:])
            return data["image"], labels

        return data["image"]

    @staticmethod
    def compute_labels_weights(labels):
        weights = np.bincount(labels)
        weights[weights == 0] = 1
        weights = 1 / weights
        weights /= weights.sum()
        return weights

    @property
    def labels(self):
        return self._labels

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        img, labels = self._get_data(index)
        img, labels = self.use_transform(img, labels)
        return (self._convert_img_for_net(img), self._convert_labels_for_net(labels, img))

    @staticmethod
    def equalize_shapes(images, labels=None):
        shapes = np.stack([i.shape for i in images], 1)

        new_h = max(shapes[1])
        new_w = max(shapes[2])

        images = list(images)
        labels = list(labels) if labels is not None else labels

        for i, img in enumerate(images):
            c, h, w = img.shape

            if h == new_h and w == new_w:
                continue

            dw = (new_w - w) / 2  # width padding
            dh = (new_h - h) / 2  # height padding

            top = int(round(dh - 0.1))
            left = int(round(dw - 0.1))

            new_img = torch.full([c, new_h, new_w], 0.5, dtype=torch.float32)
            new_img[:, top : top + h, left : left + w] = img
            images[i] = new_img

            if labels is not None and len(labels[i]):
                # Normalize coordinates 0 - 1
                labels[i][:, (2, 4)] *= w / new_w  # width
                labels[i][:, (3, 5)] *= h / new_h  # height

                labels[i][:, 2] += dw / new_w  # width
                labels[i][:, 3] += dh / new_h  # height

        if labels is not None:
            return torch.stack(images, 0).float(), torch.cat(labels, 0).float()

        return torch.stack(images, 0).float()

    @staticmethod
    def collate_fn(batch):
        img, label = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()

        return CSVDataset.equalize_shapes(img, label)


class CSVDatasetValidate(CSVDataset):
    def __init__(self, path, transform=None):
        super().__init__(path, transform)
        df = pd.read_csv(path)

        dirpath, coco_path = os.path.split(path)
        coco_path = os.path.join(dirpath, "coco_" + coco_path)
        coco_path, _ = os.path.splitext(coco_path)
        coco_path += ".json"
        if not os.path.exists(coco_path):
            coco = coco_helper.dataset_from_df(df, [(i, i) for i in self._labels])

            with open(coco_path, "w") as file:
                json.dump(coco, file)
        self.coco = COCO(coco_path)


class CSVDatasetInference(CSVDataset):
    def __init__(self, path, transform=None):
        super().__init__(path, transform)

    def __getitem__(self, index):
        img0, labels = self._get_data(index)
        img = self.use_transform(img0)
        return self._convert_img_for_net(img), labels, img0

    @staticmethod
    def collate_fn(batch):
        img, label, img0 = list(zip(*batch))
        img = CSVDataset.equalize_shapes(img)
        return img, label, img0
