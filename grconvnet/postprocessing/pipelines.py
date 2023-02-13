"""_summary_
"""

from typing import List, Any, Dict
from abc import ABC, abstractmethod

import torch
import numpy as np
from torchtyping import TensorType
from skimage.filters import gaussian  # pylint:disable=no-name-in-module
from nptyping import NDArray, Shape, Float
from torch.nn import Identity

from grconvnet.datatypes import ImageGrasp, RealGrasp
from grconvnet.utils.geometry import get_antipodal_points
from . import custom_transforms as CT


class PostprocessorBase(ABC):
    def __init__(self):
        self.intermediate_results: Dict[str, Any] = {}

    @abstractmethod
    def __call__(
        self, network_output: TensorType[1, 4, 1, 224, 224]
    ) -> List[ImageGrasp]:
        pass


class LegacyPostprocessor(PostprocessorBase):
    def __init__(self, blur: bool = True):
        """Postprocessing as done in the original implementation. This converts
        the output of the NN to grasp angles, scaled the width, and blurres the
        images. Results in

        Args:
            blur (bool, optional): Whether to blurr the resultign images.
                Defaults to True.
        """
        super().__init__()

        self.blur = blur

    def __call__(
        self,
        network_output: TensorType[1, 4, 1, 224, 224],
    ) -> List[ImageGrasp]:
        network_output = torch.squeeze(network_output)
        q_img, cos_img, sin_img, width_img = network_output

        q_img = q_img.cpu().numpy().squeeze()
        angle_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
        width_img = width_img.cpu().numpy().squeeze() * 150.0  # ???

        if self.blur:
            q_img = gaussian(q_img, 2.0, preserve_range=True)
            angle_img = gaussian(angle_img, 2.0, preserve_range=True)
            width_img = gaussian(width_img, 1.0, preserve_range=True)

        self.intermediate_results["q_img"] = q_img
        self.intermediate_results["angle_img"] = angle_img
        self.intermediate_results["width_img"] = width_img

        # no conversion to ImageGrasp as this was not part of the original
        # returning empty list for consistency
        return []


class Postprocessor(PostprocessorBase):
    def __init__(
        self,
        blur: bool = True,
        width_scale: int = 150,
        min_distance_between_grasps: int = 20,
        quality_threshold: float = 0.2,
        n_grasps: int = 2,
    ):
        super().__init__()

        # TODO refactor in a modular way (subcompoents as init arguments)

        self.blur = blur

        self.angle_converter = CT.AngleConverter()
        self.blurrer_s1 = CT.SkGaussian(1)
        self.blurrer_s2 = CT.SkGaussian(2)
        self.width_scaler = CT.Scaler(width_scale)
        # self.blurrer = T.GaussianBlur(kernel_size)

        self.localizer = CT.GraspLocalizer(
            min_distance=min_distance_between_grasps,
            threshold=quality_threshold,
            n_grasps=n_grasps,
        )

    def __call__(
        self,
        network_output: TensorType[1, 4, 1, 224, 224],
    ) -> List[ImageGrasp]:
        # we have only a bath size of 1 and all "images" have a single channel
        network_output = torch.squeeze(network_output)
        # many modules of the postprocessing pipeline work on numpy
        network_output = network_output.cpu().detach().numpy()

        q_img, cos_img, sin_img, width_img = network_output

        angle_img = self.angle_converter((sin_img, cos_img))
        width_img = self.width_scaler(width_img)

        if self.blur:
            q_img = self.blurrer_s2(q_img)
            angle_img = self.blurrer_s2(angle_img)
            width_img = self.blurrer_s1(width_img)

        grasp_centers = self.localizer(q_img)
        grasp_angles = angle_img[tuple(grasp_centers.T)]
        grasp_widths = width_img[tuple(grasp_centers.T)]
        grasp_qualities = q_img[tuple(grasp_centers.T)]

        grasps_np = np.concatenate(
            [
                grasp_centers,
                grasp_qualities[:, np.newaxis],
                grasp_angles[:, np.newaxis],
                grasp_widths[:, np.newaxis],
            ],
            axis=1,
        )

        grasps = [
            ImageGrasp(
                center=np.array((g[1], g[0])), angle=g[3], width=g[4], quality=g[2]
            )
            for g in grasps_np
        ]

        self.intermediate_results["q_img"] = q_img
        self.intermediate_results["angle_img"] = angle_img
        self.intermediate_results["width_img"] = width_img

        return grasps


class Img2WorldConverter:  # TODO use genreal base class
    def __init__(
        self,
        coord_converter: CT.Img2WorldCoordConverter,
        decropper: CT.Decropper = None,
        height_adjuster: CT.GraspHeightAdjuster = None,
    ) -> NDArray[Shape["3"], Float]:
        # sub converter
        self.img2world_converter = coord_converter
        self.decropper = decropper or Identity()
        self.height_adjuster = height_adjuster or Identity()

        # intermediate results
        self.intermediate_results = {}

    def _decrop_grasp(self, grasp: ImageGrasp) -> ImageGrasp:
        # first we account for the fact that the image was resized and/or cropped
        center_decropped = self.decropper(grasp.center)

        p1, p2 = get_antipodal_points(grasp.center, -grasp.angle, grasp.width)
        width_decropped = np.linalg.norm(self.decropper(p1) - self.decropper(p2))
        grasp_decropped = ImageGrasp(
            center_decropped, grasp.quality, grasp.angle, width_decropped
        )

        return grasp_decropped

    def _get_center_depth(self, grasp_decropped: ImageGrasp, orig_depth_image):
        orig_depth_image = orig_depth_image.squeeze().numpy()
        center_depth = orig_depth_image[
            int(grasp_decropped.center[1]), int(grasp_decropped.center[0])
        ]

        return center_depth

    def _get_width_world(self, grasp_decropped: ImageGrasp, center_depth: float):
        antipodal_points_img = get_antipodal_points(
            grasp_decropped.center, -grasp_decropped.angle, grasp_decropped.width
        )

        antipodal_points_world = [
            self.img2world_converter(p, center_depth) for p in antipodal_points_img
        ]

        width_world = np.linalg.norm(
            antipodal_points_world[0] - antipodal_points_world[1]
        )

        return width_world

    def _get_angle_world(self, grasp_decropped: ImageGrasp, center_depth: float):
        antipodal_points_img = get_antipodal_points(
            grasp_decropped.center, -grasp_decropped.angle, grasp_decropped.width
        )

        antipodal_points_world = [
            self.img2world_converter(p, center_depth) for p in antipodal_points_img
        ]
        self.intermediate_results["antipodal_points_world"] = antipodal_points_world

        angle_world = np.arctan2(
            antipodal_points_world[0][1] - antipodal_points_world[1][1],
            antipodal_points_world[0][0] - antipodal_points_world[1][0],
        )

        # bring angle_world into the range [-pi/2, pi/2]
        if angle_world > np.pi / 2:
            angle_world -= np.pi
        elif angle_world < -np.pi / 2:
            angle_world += np.pi

        return angle_world

    def __call__(
        self,
        grasp: ImageGrasp,
        orig_depth_image: TensorType[1, "H, W"],
    ) -> RealGrasp:
        # if not isinstance(self.decropper, Identity):
        #     assert tuple(orig_depth_image.shape[1:]) == self.decropper.original_img_size

        grasp_decropped = self._decrop_grasp(grasp)

        # convert the grasp center to world frame
        center_depth = self._get_center_depth(grasp_decropped, orig_depth_image)
        center_world = self.img2world_converter(grasp_decropped.center, center_depth)

        # convert the width of the grasp
        width_world = self._get_width_world(grasp_decropped, center_depth)

        # convert the angle
        angle_world = self._get_angle_world(grasp_decropped, center_depth)

        grasp_world = RealGrasp(
            center_world,
            grasp.quality,
            angle_world,
            width_world,
        )

        self.intermediate_results["grasp_decropped"] = grasp_decropped
        self.intermediate_results["center_depth"] = center_depth
        self.intermediate_results["grasp_world_raw"] = grasp_world

        grasp_world = self.height_adjuster(grasp_world)

        return grasp_world
