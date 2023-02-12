from pathlib import Path
from typing import Union, Callable

import numpy as np
import torch
import open3d as o3d
from torchtyping import TensorType
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from scipy.spatial.transform import Rotation

from ..datatypes import CameraData

# Datasets __getitem__ should alway return CameraData object which contains rgb, depth and points
# additional fields can be added to the CameraData (or subclass) object
# this allows us to use the same dataloader for all datasets
# therfore all necessary conversions should be done in the __getitem__ method in order
# to make the return value of the __getitem__ method compatible with the preprocessing pipeline
# and the dataloader


class CornellDataset(Dataset):
    def __init__(self, root_dir: Path, transform: Callable = None):
        """Initializes torch dataset class for the CornellGraspDataset.
        The dataset is index from 0 to 745. So you must substract 100 from the
        file names to obtain the index.

        Args:
            root_dir (Path): Location of the dataset.
            transform (Callable, optional): Preprocessing pipeline to apply to each
                sample in the dataset when accessed. Defaults to None.
        """
        self.root_dir = Path(root_dir)

        self.transform = transform

    def __len__(self):
        length = 0

        for subdir in self.root_dir.iterdir():
            if subdir.name in (
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
            ):
                length += int(len(list(subdir.glob("*"))) / 5)

        return length

    def get_item_base_path(self, index: int) -> str:
        if index >= 850:
            index = index + 50
        folder = f"{index // 100 + 1:02}"
        file_index = f"{index % 100:02}"

        item_base_path = self.root_dir / folder / f"pcd{folder}{file_index}"

        return str(item_base_path)

    def get_rgb(self, index: int) -> TensorType[3, 480, 640]:
        rgb = Image.open(f"{self.get_item_base_path(index)}r.png", "r")
        # rgb = io.imread(f"{self.get_item_base_path(index)}r.png")
        rgb = F.pil_to_tensor(rgb)
        # rgb = rgb.float()
        return rgb

    def get_depth(self, index: int) -> TensorType[1, 480, 640]:
        depth = Image.open(f"{self.get_item_base_path(index)}d.tiff", "r")
        # depth = io.imread(f"{self.get_item_base_path(index)}d.tiff")
        depth = F.pil_to_tensor(depth)
        return depth

    def _load_grasps_file(self, path: Union[str, Path]) -> TensorType["n_grasps", 4, 2]:
        grasp_data = np.loadtxt(path)
        assert len(grasp_data) % 4 == 0

        grasp_rectangles = grasp_data.reshape(int(len(grasp_data) / 4), 4, 2)

        return torch.from_numpy(grasp_rectangles)

    def get_pos_grasps(self, index: int) -> TensorType["n_grasps", 4, 2]:
        return self._load_grasps_file(f"{self.get_item_base_path(index)}cpos.txt")

    def get_neg_grasps(self, index: int) -> TensorType["n_grasps", 4, 2]:
        return self._load_grasps_file(f"{self.get_item_base_path(index)}cneg.txt")

    def get_point_cloud(self, index: int) -> TensorType["n_grasps", 4, 2]:
        pcd = o3d.io.read_point_cloud(
            f"{self.get_item_base_path(index)}.txt", format="pcd"
        )
        # pcd.points is of vector type, converting this type to Tensor is very slow
        # therfore we use np.asarray as a intermediate step
        return torch.tensor(np.asarray(pcd.points))

    def get_name(self, index: int) -> str:
        return Path(self.get_item_base_path(index)).parts[-1]

    def get_segmentation(self, index: int):  # -> TensorType[1, 480, 640]:
        size = [1, *self.get_rgb(index).size()[1:]]
        segmentation = torch.ones(size)
        return segmentation

    def __getitem__(self, index):
        sample = CameraData(
            rgb=self.get_rgb(index),
            depth=self.get_depth(index),
            points=self.get_point_cloud(index),
            segmentation=self.get_segmentation(index),
            pos_grasps=self.get_pos_grasps(index),
            neg_grasps=self.get_neg_grasps(index),
            name=self.get_name(index),
        )
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class YCBSimulationData(Dataset):
    def __init__(self, root_dir: Path, transform: Callable = None):
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(list(self.root_dir.glob("*.npz")))

    def __getitem__(self, index: int):
        all_sample_names = [
            p.parts[-1] for p in self.root_dir.iterdir() if not p.is_dir()
        ]

        all_sample_names = sorted(all_sample_names)
        sample_name = all_sample_names[index]
        sample_path = self.root_dir / sample_name

        simulation_data = np.load(sample_path)

        sample = CameraData(
            rgb=torch.from_numpy(simulation_data["rgb_img"]).permute((2, 0, 1)),
            depth=torch.unsqueeze(torch.from_numpy(simulation_data["depth_img"]), 0),
            points=torch.from_numpy(simulation_data["point_cloud"]),
            segmentation=torch.unsqueeze(
                torch.from_numpy(simulation_data["seg_img"].astype("uint8")), 0
            ),
            cam_intrinsics=simulation_data["cam_intrinsics"],
            cam_pos=simulation_data["cam_pos"],
            cam_rot=Rotation.from_quat(
                simulation_data["cam_quat"][[3, 0, 1, 2]]
            ).as_matrix(),
            name=sample_name.split(".")[0],
        )

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
