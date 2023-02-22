"""_summary_
"""

import torch
from torchvision import transforms as T
from torchtyping import TensorType


class FlattenedNormalize:
    def __init__(self, std: int = 255):
        super().__init__()
        self.std = std

    def __call__(
        self, img: TensorType["n_channels", "h", "w"]
    ) -> TensorType["n_channels", "h", "w"]:
        return (img - torch.mean(img.float())) / self.std


class SquareCrop:
    def __call__(
        self, img: TensorType["c", "h", "w"]
    ) -> TensorType["c", "min(h,c)", "min(h,c)"]:
        min_dim = min(img.shape[1:])
        return T.CenterCrop(size=min_dim)(img)


class CenterCropResized:
    def __init__(self, size: int = (224, 224)):
        self.pipeline = T.Compose([SquareCrop(), T.Resize(size)])

    def __call__(
        self, img: TensorType["c", "h", "w"]
    ) -> TensorType["c", "size", "size"]:
        return self.pipeline(img)


class Masker:
    def __init__(
        self, negative_value: TensorType["c"], positive_value: TensorType["c"] = None
    ):
        super().__init__()
        self.negative_value = torch.tensor(negative_value)
        self.positive_value = torch.tensor(positive_value)

    def _make_constant_image(
        self, img: TensorType["c", "h", "w"], value: TensorType["c"]
    ) -> TensorType["c", "h", "w"]:
        constant_img = value.repeat(img.shape[1:]).reshape(
            (img.shape[1], img.shape[2], img.shape[0])
        )
        constant_img = constant_img.permute(2, 0, 1)
        return constant_img

    def __call__(
        self, img: TensorType["c", "h", "w"], mask: TensorType[1, "h", "w"]
    ) -> TensorType["c", "h", "w"]:
        assert img.shape[1:] == mask.shape[1:]

        mask = mask.squeeze()

        if self.positive_value is None:
            postive_img = img
        else:
            postive_img = self._make_constant_image(img, self.positive_value)

        if self.negative_value is None:
            negative_img = img
        else:
            negative_img = self._make_constant_image(img, self.negative_value)

        masked_img = torch.where(mask == 1, postive_img, negative_img)

        return masked_img
