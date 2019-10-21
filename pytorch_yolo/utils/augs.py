import cv2 as cv
import numpy as np

from albumentations import DualTransform, denormalize_bbox, normalize_bbox


class LetterBox(DualTransform):
    def get_params_dependent_on_targets(self, params):
        super(LetterBox, self).get_params_dependent_on_targets(params)

    def __init__(
        self,
        new_shape=416,
        border_mode=cv.BORDER_REPLICATE,
        border_color=None,
        always_apply=True,
        p=1,
        interpolation=cv.INTER_AREA,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.new_shape = new_shape
        self.border_color = border_color
        self.border_mode = border_mode
        self.interpolation = interpolation

    def update_params(self, params, **kwargs):
        params = super().update_params(params, **kwargs)
        h = params["rows"]
        w = params["cols"]

        if isinstance(self.new_shape, int):  # rectangle
            r = h / w
            shape = [1, 1]
            if r < 1:
                shape = [r, 1]
            elif r > 1:
                shape = [1, 1 / r]

            target_shape = np.ceil(np.array(shape) * self.new_shape / 32).astype(np.int) * 32
        else:
            target_shape = self.new_shape

        ratio = max(target_shape) / max(h, w)
        target_height, target_width = target_shape

        resize_height = int(round(h * ratio))
        resize_width = int(round(w * ratio))

        # Compute padding https://github.com/ultralytics/yolov3/issues/232
        if isinstance(self.new_shape, int):  # rectangle
            pad_left = np.mod(target_width - resize_width, 32) / 2
            pad_top = np.mod(target_height - resize_height, 32) / 2
        else:
            pad_left = (target_width - resize_width) / 2
            pad_top = (target_height - resize_height) / 2

        pad_left = int(pad_left)
        pad_top = int(pad_top)
        pad_right = target_width - resize_width - pad_left
        pad_bottom = target_height - resize_height - pad_top

        params.update(
            {
                "pad_left": pad_left,
                "pad_top": pad_top,
                "pad_right": pad_right,
                "pad_bottom": pad_bottom,
                "resize_ratio": ratio,
            }
        )
        return params

    def apply(self, img, pad_left, pad_right, pad_top, pad_bottom, resize_ratio, **params):
        img = cv.resize(
            img, None, fx=resize_ratio, fy=resize_ratio, interpolation=self.interpolation
        )  # resized, no border
        img = cv.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right, self.border_mode, value=self.border_color
        )  # padded square

        return img

    def apply_to_bbox(self, bbox, pad_left, pad_right, pad_top, pad_bottom, resize_ratio, cols, rows, **params):
        x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)
        bbox = [
            x_min * resize_ratio + pad_left,
            y_min * resize_ratio + pad_top,
            x_max * resize_ratio + pad_left,
            y_max * resize_ratio + pad_top,
        ]
        return normalize_bbox(
            bbox, rows * resize_ratio + pad_top + pad_bottom, cols * resize_ratio + pad_left + pad_right
        )

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def get_transform_init_args_names(self):
        return "new_shape", "border_color", "border_mode", "interpolation"
