"""_summary_
"""

from abc import abstractmethod
from typing import Any, Dict, Union, Iterable
from collections import deque

from torchvision import transforms as T
import torch
from torchtyping import TensorType
import numpy as np

# from grconvnet._orig.utils.data.camera_data import CameraData as CameraDataLegacy
from grconvnet.datatypes import CornellData, DatasetPoint
from . import custom_transforms as CT


class PreprocessorBase:
    def __init__(self, intermediate_results_queue_size: int):
        self.intermediate_results: Iterable[Dict[str, Any]] = deque(
            maxlen=intermediate_results_queue_size
        )

    @abstractmethod
    def __call__(self, sample: DatasetPoint) -> TensorType[4, 224, 224]:
        pass


# class LegacyPreprocessor(PreprocessorBase):
#     """Preprocesses a rgb image and depth image in exactly the same way the original
#     imlpementation does it. Also uses the same helper classes etc.
#     """

#     def __call__(self, sample: CornellData) -> TensorType[4, 224, 224]:
#         # the camera data object expsects numpy arrays for its method calls!!!
#         orig_resizer = CameraDataLegacy(include_depth=True, include_rgb=True)

#         # (1, 4, 224, 224)
#         input_tensor, _, _ = orig_resizer.get_data(
#             rgb=np.array(sample.rgb).transpose((1, 2, 0)),
#             depth=np.array(sample.depth).transpose((1, 2, 0)),
#         )

#         input_tensor = input_tensor.squeeze(0)  # (4, 224, 224)

#         return input_tensor


class RebuildLegacyPreprocessor(PreprocessorBase):
    def __init__(self, intermediate_results_queue_size: int = 1):
        super().__init__(intermediate_results_queue_size)

    def __call__(self, sample: CornellData) -> TensorType[4, 224, 224]:
        rgb_np = np.array(sample.rgb).transpose((1, 2, 0))  # (480, 640, 3)
        depth_np = np.array(sample.depth).transpose((1, 2, 0))  # (480, 640, 1)

        # crop
        # (224,224,3)
        rgb_cropped = rgb_np[
            (480 - 224) // 2 : (480 + 224) // 2, (640 - 224) // 2 : (640 + 224) // 2
        ]
        # (224,224,1)
        depth_cropped = depth_np[
            (480 - 224) // 2 : (480 + 224) // 2, (640 - 224) // 2 : (640 + 224) // 2
        ]
        seg_cropped = None

        rgb_masked = rgb_cropped.copy()

        # normalize
        rgb_norm = rgb_cropped.astype(np.float32) / 255.0
        rgb_norm -= rgb_norm.mean()
        rgb_norm = rgb_norm.transpose((2, 0, 1))  # (3, 224, 224)

        depth_norm = depth_cropped.astype(np.float32) / 255.0
        depth_norm -= depth_norm.mean()
        depth_norm = depth_norm.transpose((2, 0, 1))  # (1, 480, 640)

        # (4,480,640)
        input_tensor = np.concatenate((depth_norm, rgb_norm), 0)

        self.intermediate_results.append({})
        self.intermediate_results[-1]["rgb_cropped"] = rgb_cropped
        self.intermediate_results[-1]["depth_cropped"] = depth_cropped
        self.intermediate_results[-1]["seg_cropped"] = seg_cropped
        self.intermediate_results[-1]["rgb_masked"] = rgb_masked
        self.intermediate_results[-1]["rgb_norm"] = rgb_norm
        self.intermediate_results[-1]["depth_norm"] = depth_norm

        return torch.from_numpy(input_tensor)


class Preprocessor(PreprocessorBase):
    def __init__(
        self,
        reformatter: Union[T.CenterCrop, CT.CenterCropResized],
        masker: CT.Masker,
        intermediate_results_queue_size: int = 1,
    ):
        """_summary_

        Args:
            resize (bool, optional): If resize=True we first square crop the image
                and then scale the image to 224. Otherwise we only centercrop the image.
                Defaults to False.
            mask_rgb_neg_color (TensorType[&quot;3&quot;], optional): _description_. Defaults to None.
            mask_rgb_pos_color (TensorType[&quot;3&quot;], optional): _description_. Defaults to None.
        """
        super().__init__(intermediate_results_queue_size)

        self.masker = masker or (lambda rgb, seg: rgb)
        self.reformatter = reformatter
        self.normalizer = CT.FlattenedNormalize(255)

    def __call__(self, sample: DatasetPoint) -> TensorType[4, 224, 224]:
        rgb_cropped = self.reformatter(sample.rgb)
        depth_cropped = self.reformatter(sample.depth)
        seg_cropped = self.reformatter(sample.segmentation)

        # if the masking values are None the masker will not mask anything
        rgb_masked = self.masker(rgb_cropped, seg_cropped)

        rgb_norm = self.normalizer(rgb_masked)
        depth_norm = self.normalizer(depth_cropped)

        input_tensor = torch.cat([depth_norm, rgb_norm], dim=0)

        self.intermediate_results.append({})
        self.intermediate_results[-1]["initial_sample"] = sample
        self.intermediate_results[-1]["rgb_cropped"] = rgb_cropped
        self.intermediate_results[-1]["depth_cropped"] = depth_cropped
        self.intermediate_results[-1]["seg_cropped"] = seg_cropped
        self.intermediate_results[-1]["rgb_masked"] = rgb_masked
        self.intermediate_results[-1]["rgb_norm"] = rgb_norm
        self.intermediate_results[-1]["depth_norm"] = depth_norm

        return input_tensor
