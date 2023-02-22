from dataclasses import dataclass
from abc import ABC

from nptyping import NDArray, Float, Shape
from torchtyping import TensorType
import torch

# here we define the datatypes we use in the project
# theses should contain only data and no logic as we want to use the dataloaders and pipelines
# for conversions etc
# so classmethods in the style of .from_xy() etc should be avoided


@dataclass
class DatasetPoint(ABC):
    name: str


@dataclass
class YCBData(DatasetPoint):
    rgb: TensorType[3, "h", "w", torch.uint8]
    depth: TensorType[1, "h", "w", torch.float32]
    points: TensorType["n_points", 3]
    segmentation: TensorType[1, "h", "w", torch.uint8]
    cam_intrinsics: NDArray[Shape["3, 3"], Float] = None
    cam_pos: NDArray[Shape["3"], Float] = None
    cam_rot: NDArray[Shape["4"], Float] = None


@dataclass
class CornellData(DatasetPoint):
    rgb: TensorType[3, "h", "w", torch.uint8]
    depth: TensorType[1, "h", "w", torch.float32]
    points: TensorType["n_points", 3]
    segmentation: TensorType[1, "h", "w", torch.uint8]
    pos_grasps: TensorType["n_pos_grasps", 4, 2] = None
    neg_grasps: TensorType["n_pos_grasps", 4, 2] = None


@dataclass
class ImageGrasp:
    center: NDArray[
        Shape["2"], Float
    ]  # (x, y) coordinates when viewing/displaying the image (= (col, row) in image marix)
    quality: float
    angle: float
    width: float
    # center (NDArray[Shape[&quot;2&quot;], Float]): (x, y) coordinates when
    #     viewing/displaying the image. This is NOT the the pixel at img[center[0], center[1]]
    #     but the pixel at img[center[1], center[0]].


@dataclass
class RealGrasp:
    center: NDArray[Shape["3"], Float]
    quality: float
    angle: float
    width: float
