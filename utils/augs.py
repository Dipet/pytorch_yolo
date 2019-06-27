import numpy as np

import random

import cv2 as cv


def _get_img_targets(data):
    if isinstance(data, (tuple, list)):
        return data

    return data, None


class RandomAffineYOLO:
    def __init__(self,
                 degrees=(-5, 5),
                 translate=(0.1, 0.1),
                 scale=(0.9, 1.1),
                 shear=(-2, 2),
                 border_value=(127.5, 127.5, 127.5)):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.border_value = border_value

    def __call__(self, data):
        img, targets = _get_img_targets(data)

        height, width = img.shape[:2]

        # Rotation and Scale
        rot_mat = np.eye(3)
        angle = random.uniform(self.degrees[0], self.degrees[1])
        scale = random.uniform(self.scale[0], self.scale[1])
        rot_mat[:2] = cv.getRotationMatrix2D(angle=angle,
                                              center=(width / 2, height / 2),
                                              scale=scale)

        # Translation
        trans_mat = np.eye(3)
        trans_mat[0, 2] = random.uniform(-1, 1) * self.translate[0] * height
        trans_mat[1, 2] = random.uniform(-1, 1) * self.translate[1] * width

        # Shear
        shear_mat = np.eye(3)
        shear_mat[0, 1] = np.tan(np.deg2rad(random.uniform(self.shear[0],
                                                           self.shear[1])))
        shear_mat[1, 0] = np.tan(np.deg2rad(random.uniform(self.shear[0],
                                                           self.shear[1])))

        transform_mat = shear_mat @ trans_mat @ rot_mat
        img = cv.warpPerspective(img,
                                  transform_mat,
                                  dsize=(width, height),
                                  flags=cv.INTER_LINEAR,
                                  borderValue=self.border_value)

        # Return warped points also
        if targets is not None:
            n = targets.shape[0]
            points = targets[:, 1:5].copy()
            area0 = ((points[:, 2] - points[:, 0])
                     * (points[:, 3] - points[:, 1]))

            # warp points
            xy = np.ones((n * 4, 3))
            # x1y1, x2y2, x1y2, x2y1
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
            xy = (xy @ transform_mat.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1),
                                 x.max(1), y.max(1))).reshape(4, n).T

            # reject warped points outside of image
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width - 1)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height - 1)

            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]

            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 2) & (h > 2) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, 1:5] = xy[i]

        if targets is None:
            return img

        return img, targets


class LetterBox:
    def __init__(self,
                 new_shape=416,
                 color=(127.5, 127.5, 127.5)):
        self.new_shape = new_shape
        self.color = color

    def __call__(self, data):
        img, targets = _get_img_targets(data)

        h, w = img.shape[:2]

        if isinstance(self.new_shape, int):  # rectangle
            r = h / w
            shape = [1, 1]
            if r < 1:
                shape = [r, 1]
            elif r > 1:
                shape = [1, 1 / r]

            new_shape = np.ceil(np.array(shape)
                                * self.new_shape / 32).astype(np.int) * 32
        else:
            new_shape = self.new_shape

        ratio = max(new_shape) / max(h, w)
        target_height, target_width = new_shape

        new_h = int(round(h * ratio))
        new_w = int(round(w * ratio))

        # Compute padding https://github.com/ultralytics/yolov3/issues/232
        if isinstance(self.new_shape, int):  # rectangle
            dw = np.mod(target_width - new_w, 32) / 2
            dh = np.mod(target_height - new_h, 32) / 2
        else:
            dw = (target_width - new_w) / 2
            dh = (target_height - new_h) / 2

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv.resize(img, (new_w, new_h),
                         interpolation=cv.INTER_AREA)  # resized, no border
        img = cv.copyMakeBorder(img,
                                 top, bottom,
                                 left, right,
                                 cv.BORDER_CONSTANT,
                                 value=self.color)  # padded square

        if targets is None:
            return img

        targets[:, (1, 3)] = ratio * targets[:, (1, 3)] + dw
        targets[:, (2, 4)] = ratio * targets[:, (2, 4)] + dh

        return img, targets


class RandomHSVYOLO:
    def __init__(self, fraction=0.5):
        self.fraction = fraction

    def __call__(self, data):
        img, targets = _get_img_targets(data)

        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        saturation = img_hsv[:, :, 1].astype(np.float32)
        value = img_hsv[:, :, 2].astype(np.float32)

        d_sat = random.uniform(-1, 1) * self.fraction + 1
        d_val = random.uniform(-1, 1) * self.fraction + 1

        saturation *= d_sat
        value *= d_val

        img_hsv[:, :, 1] = saturation if d_sat < 1 else saturation.clip(None,
                                                                        255)
        img_hsv[:, :, 2] = value if d_val < 1 else value.clip(None, 255)

        img = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)

        if targets is None:
            return img

        return img, targets


class RandomFlip:
    def __init__(self, x=False, y=True, target_format='xyxy'):
        if not np.isin(target_format, ['xyhw', 'xyxy']):
            raise ValueError(f'Wrong format "{format}".'
                             'Supported: ["xyxy", "xyhw"]')

        self.x = x
        self.y = y
        self.format = target_format

    def __call__(self, data):
        img, targets = _get_img_targets(data)
        shape = img.shape

        flip_code = None
        if self.y and random.random() > 0.5:
            targets = self._y_flip(targets, shape)
            flip_code = 1

        if self.x and random.random() > 0.5:
            targets = self._x_flip(targets, shape)
            flip_code = -1 if flip_code else 0

        if flip_code is not None:
            img = cv.flip(img, flip_code)

        if targets is None:
            return img

        return img, targets

    def _y_flip(self, targets: np.ndarray, shape):
        if targets is None:
            return targets

        w = 1 if bool((targets <= 1).all()) else shape[1]
        if self.format == 'xyhw':
            targets[:, 1] = w - targets[:, 1]
        elif self.format == 'xyxy':
            left = targets[:, 1].copy()
            right = targets[:, 3].copy()

            targets[:, 1] = w - right
            targets[:, 3] = w - left

        return targets

    def _x_flip(self, targets: np.ndarray, shape):
        if targets is None:
            return targets

        h = 1 if bool((targets <= 1).all()) else shape[0]
        if self.format == 'xyhw':
            targets[:, 2] = h - targets[:, 2]
        elif self.format == 'xyxy':
            top = targets[:, 2].copy()
            bottom = targets[:, 4].copy()

            targets[:, 2] = h - bottom
            targets[:, 4] = h - top

        return targets


class CheckChannels:
    def __init__(self, out_channels):
        self.out_channels = out_channels

    def __call__(self, data):
        img, targets = _get_img_targets(data)

        if len(img.shape) == 2 and self.out_channels != 1:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        elif img.shape[-1] != self.out_channels:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)

        if targets is None:
            return img

        return img, targets


class ExtractColor:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __call__(self, data):
        img, targets = _get_img_targets(data)

        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        mask = cv.inRange(img, self.lower, self.upper)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        mask = cv.dilate(mask, kernel, iterations=2)

        img = cv.bitwise_and(img, img, mask=mask)
        img = cv.cvtColor(img, cv.COLOR_HSV2BGR)

        if targets is None:
            return img

        return img, targets


if __name__ == '__main__':
    img = cv.imread('/home/druzhinin/Datasets/BMP/BMP_1/000700.raw.jpg',
                     cv.IMREAD_COLOR)
    ltb = LetterBox()
    img = ltb(img)

    print(img.shape)

    import matplotlib.pyplot as plt

    plt.imshow(img)
    plt.show()
