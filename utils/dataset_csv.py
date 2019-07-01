import os
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

import pandas as pd

from multiprocessing import Value

import torch

try:
    from utils import coco_helper
    from utils import xyxy2xywh
    from augs import LetterBox, CheckChannels
except:
    from . import coco_helper
    from .utils import xyxy2xywh
    from .augs import LetterBox, CheckChannels

import json

from pycocotools.coco import COCO

from torchvision.transforms import Compose


class CSVDataset(Dataset):  # for training/testing
    def __init__(self,
                 path,
                 img_size=416,
                 transform=None,
                 in_channels=3,):
        df = pd.read_csv(path)
        if df['type'].dtype == np.object:
            map_type = {j: i for i, j in enumerate(sorted(df['type'].unique()))}
            df['type'] = df['type'].replace(map_type)

        if min(df['type']) > 0: # Fix class
            df['type'] -= 1
        self.cls_number = len(df['type'].unique())
        self.dataset = {}
        self.class_weight = self.compute_labels_weights(df['type'])
        self.letterbox = LetterBox(img_size)
        self.check_channels = CheckChannels(in_channels)


        dirpath, name = os.path.split(path)
        labels_path = os.path.join(dirpath, 'labels_' + name)
        if os.path.exists(labels_path):
            self._labels = pd.read_csv(labels_path)['label'].values
        else:
            self._labels = np.array([str(i) for i in  sorted(df['type'].unique())])

        for _, row in tqdm(df.iterrows(), total=len(df), desc='Parse csv file'):
            path = row['image_path']

            label = [row['type'],
                     row['left'],
                     row['top'],
                     row['right'],
                     row['bottom']]

            if path in self.dataset:
                self.dataset[path].append(label)
            else:
                self.dataset[path] = [label]

        for i in list(self.dataset.keys()):
            self.dataset[i] = np.array(self.dataset[i]).astype(float)

        n = len(self.dataset)
        assert n > 0, 'No images found in %s' % path
        self._img_size = Value('i', img_size)

        self.img_files = list(self.dataset.keys())

        self.transform = transform

    def use_transform(self, img, labels):
        img, labels = self.letterbox((img, labels))
        if self.transform:
            img, labels = self.transform((img, labels))
        return self.check_channels((img, labels))

    def compute_labels_weights(self, labels):
        weights = np.bincount(labels)
        weights[weights == 0] = 1
        weights = 1 / weights
        weights /= weights.sum()
        return weights

    @property
    def img_size(self):
        return self._img_size.value

    @img_size.setter
    def img_size(self, size):
        self._img_size.value = size

    @property
    def labels(self):
        return self._labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        labels = self.dataset[img_path].copy()

        img = cv.imread(img_path, cv.IMREAD_COLOR)  # BGR
        assert img is not None, 'File Not Found ' + img_path

        labels[:, 1] = np.maximum(labels[:, 1], 0)
        labels[:, 2] = np.maximum(labels[:, 2], 0)
        labels[:, 3] = np.minimum(labels[:, 3], img.shape[1] - 1)
        labels[:, 4] = np.minimum(labels[:, 4], img.shape[0] - 1)

        img, labels = self.use_transform(img, labels)

        # Convert labels format
        h, w = img.shape[:2]
        labels[:, 1:] = xyxy2xywh(labels[:, 1:])
        labels[:, (1, 3)] = labels[:, (1, 3)] / w
        labels[:, (2, 4)] = labels[:, (2, 4)] / h

        # Normalize 0-1
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255

        # Add image index axis
        result_labels = np.zeros((len(labels), 6), dtype=np.float32)
        if len(labels):
            result_labels[:, 1:] = labels

        # To Pytorch format
        img = np.transpose(img, (2, 0, 1))

        return torch.from_numpy(img), torch.from_numpy(result_labels)

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
            new_img[:, top:top + h, left:left + w] = img
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
    def __init__(self,
                 path,
                 img_size=416,
                 transform=None,
                 **kwargs):
        super().__init__(path, img_size, transform, **kwargs)
        df = pd.read_csv(path)

        dirpath, coco_path = os.path.split(path)
        coco_path = os.path.join(dirpath, 'coco_' + coco_path)
        coco_path, _ = os.path.splitext(coco_path)
        coco_path += '.json'
        if not os.path.exists(coco_path):
            coco = coco_helper.dataset_from_df(df, [(i, i) for i in self._labels])

            with open(coco_path, 'w') as file:
                json.dump(coco, file)
        self.coco = COCO(coco_path)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        labels = self.dataset[img_path].copy()

        img = cv.imread(img_path, cv.IMREAD_COLOR)  # BGR
        orig_shape = img.shape[:2]
        assert img is not None, 'File Not Found ' + img_path

        labels[:, 1] = np.maximum(labels[:, 1], 0)
        labels[:, 2] = np.maximum(labels[:, 2], 0)
        labels[:, 3] = np.minimum(labels[:, 3], img.shape[1] - 1)
        labels[:, 4] = np.minimum(labels[:, 4], img.shape[0] - 1)

        img, labels = self.use_transform(img, labels)

        # Normalize 0-1
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255

        # To Pytorch format
        img = np.transpose(img, (2, 0, 1))
        result_labels = np.zeros((len(labels), 6))
        result_labels[:, 1:] = labels

        return (torch.from_numpy(img), result_labels, img_path, orig_shape)

    @staticmethod
    def collate_fn(batch):
        img, label, img_path, orig_shape = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()

        img = CSVDataset.equalize_shapes(img)
        return img, np.concatenate(label), img_path, orig_shape


class CSVDatasetInference(CSVDataset):
    def __init__(self,
                 path,
                 img_size=416,
                 transform=None,
                 **kwargs):
        super().__init__(path, img_size, transform, **kwargs)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        labels = self.dataset[img_path].copy()

        img = cv.imread(img_path, cv.IMREAD_COLOR)  # BGR
        img0 = img.copy()
        assert img is not None, 'File Not Found ' + img_path

        labels[:, 1] = np.maximum(labels[:, 1], 0)
        labels[:, 2] = np.maximum(labels[:, 2], 0)
        labels[:, 3] = np.minimum(labels[:, 3], img.shape[1] - 1)
        labels[:, 4] = np.minimum(labels[:, 4], img.shape[0] - 1)

        img, labels = self.use_transform(img, labels)

        # Normalize 0-1
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255

        # To Pytorch format
        img = np.transpose(img, (2, 0, 1))

        return torch.from_numpy(img), labels, img0

    @staticmethod
    def collate_fn(batch):
        img, label, img0 = list(zip(*batch))  # transposed

        img = CSVDataset.equalize_shapes(img)
        return img, label, img0


class CSVDatasetVideo(Dataset):  # for training/testing
    def __init__(self,
                 path,
                 img_size=416,
                 transform=None,
                 in_channels=3,):
        df = pd.read_csv(path)
        if df['type'].dtype == np.object:
            map_type = {j: i for i, j in enumerate(sorted(df['type'].unique()))}
            df['type'] = df['type'].replace(map_type)

        if min(df['type']) > 0: # Fix class
            df['type'] -= 1
        self.cls_number = len(df['type'].unique())
        self.dataset = {}
        self.class_weight = self.compute_labels_weights(df['type'])
        self.letterbox = LetterBox((img_size, img_size))
        self.check_channels = CheckChannels(in_channels)

        dirpath, name = os.path.split(path)
        labels_path = os.path.join(dirpath, 'labels_' + name)
        if os.path.exists(labels_path):
            self._labels = pd.read_csv(labels_path)['label'].values
        else:
            self._labels = np.array([str(i) for i in  sorted(df['type'].unique())])

        for _, row in tqdm(df.iterrows(), total=len(df), desc='Parse csv file'):
            path = row['image_path']

            label = [row['type'],
                     row['left'],
                     row['top'],
                     row['right'],
                     row['bottom']]

            if path in self.dataset:
                self.dataset[path].append(label)
            else:
                self.dataset[path] = [label]

        for i in list(self.dataset.keys()):
            self.dataset[i] = np.array(self.dataset[i]).astype(float)

        n = len(self.dataset)
        assert n > 0, 'No images found in %s' % path

        self.img_files = df['image_path'].unique()

        self.transform = transform
        self.prev = ''
        self.subtractor = self.get_subtractor()
        self.kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

        if isinstance(self.transform, Compose):
            for i in self.transform.transforms:
                i.random_call = True
        else:
            self.transform.random_call = True

    @staticmethod
    def get_subtractor():
        return cv.createBackgroundSubtractorMOG2(10, 4, detectShadows=False)

    def use_transform(self, img, labels):
        img, labels = self.letterbox((img, labels))
        if self.transform:
            img, labels = self.transform((img, labels))

        mask = self.subtractor.apply(img)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, self.kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, self.kernel)

        img = np.dstack([img, mask])

        return img, labels

    def compute_labels_weights(self, labels):
        weights = np.bincount(labels)
        weights[weights == 0] = 1
        weights = 1 / weights
        weights /= weights.sum()
        return weights

    @property
    def labels(self):
        return self._labels

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]

        dir, _ = os.path.split(img_path)
        if self.prev != dir:
            self.subtractor = self.get_subtractor()

            if isinstance(self.transform, Compose):
                for i in self.transform.transforms:
                    i.random()
            else:
                self.transform.random()

        labels = self.dataset[img_path].copy()

        img = cv.imread(img_path, cv.IMREAD_COLOR)  # BGR
        assert img is not None, 'File Not Found ' + img_path

        labels[:, 1] = np.maximum(labels[:, 1], 0)
        labels[:, 2] = np.maximum(labels[:, 2], 0)
        labels[:, 3] = np.minimum(labels[:, 3], img.shape[1] - 1)
        labels[:, 4] = np.minimum(labels[:, 4], img.shape[0] - 1)

        img, labels = self.use_transform(img, labels)

        # Convert labels format
        h, w = img.shape[:2]
        labels[:, 1:] = xyxy2xywh(labels[:, 1:])
        labels[:, (1, 3)] = labels[:, (1, 3)] / w
        labels[:, (2, 4)] = labels[:, (2, 4)] / h

        # Normalize 0-1
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255

        # Add image index axis
        result_labels = np.zeros((len(labels), 6), dtype=np.float32)
        if len(labels):
            result_labels[:, 1:] = labels

        # To Pytorch format
        img = np.transpose(img, (2, 0, 1))

        return torch.from_numpy(img), torch.from_numpy(result_labels)

    @staticmethod
    def collate_fn(batch):
        img, label = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()

        return CSVDataset.equalize_shapes(img, label)
