"""_summary_
"""

from typing import List, Union, Tuple, Dict, Iterable, Any
from abc import abstractmethod
from collections import deque

import torch
import numpy as np
from torchtyping import TensorType
from skimage.filters import gaussian  # pylint:disable=no-name-in-module
from nptyping import NDArray, Shape, Float

from grconvnet.datatypes import ImageGrasp, RealGrasp
from grconvnet.utils.geometry import get_antipodal_points
from . import custom_transforms as CT


class PostprocessorBase:
    def __init__(self, intermediate_results_queue_size: int):
        self.intermediate_results: Iterable[Dict[str, Any]] = deque(
            maxlen=intermediate_results_queue_size
        )

    @abstractmethod
    def __call__(
        self, network_output: TensorType[1, 4, 1, 224, 224]
    ) -> List[ImageGrasp]:
        pass


class LegacyPostprocessor(PostprocessorBase):
    def __init__(self, blur: bool = True, intermediate_results_queue_size: int = 1):
        """Postprocessing as done in the original implementation. This converts
        the output of the NN to grasp angles, scaled the width, and blurres the
        images. Results in

        Args:
            blur (bool, optional): Whether to blurr the resultign images.
                Defaults to True.
        """
        super().__init__(intermediate_results_queue_size)

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

        self.intermediate_results.append({})
        self.intermediate_results[-1]["q_img"] = q_img
        self.intermediate_results[-1]["angle_img"] = angle_img
        self.intermediate_results[-1]["width_img"] = width_img

        # no conversion to ImageGrasp as this was not part of the original
        # returning empty list for consistency
        return []


class Postprocessor(PostprocessorBase):
    def __init__(
        self,
        q_blurrer: CT.SkGaussian,
        angle_blurrer: CT.SkGaussian,
        width_blurrer: CT.SkGaussian,
        width_scaler: CT.Scaler,
        grasp_localizer: CT.GraspLocalizer,
        intermediate_results_queue_size: int = 1,
    ):
        super().__init__(intermediate_results_queue_size)

        self.angle_converter = CT.AngleConverter()

        self.q_blurrer = q_blurrer or (lambda x: x)
        self.width_blurrer = width_blurrer or (lambda x: x)
        self.angle_blurrer = angle_blurrer or (lambda x: x)

        self.width_scaler = width_scaler
        self.grasp_localizer = grasp_localizer

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

        q_img = self.q_blurrer(q_img)
        angle_img = self.angle_blurrer(angle_img)
        width_img = self.width_blurrer(width_img)

        grasp_centers = self.grasp_localizer(q_img)
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

        self.intermediate_results.append({})
        self.intermediate_results[-1]["q_img"] = q_img
        self.intermediate_results[-1]["angle_img"] = angle_img
        self.intermediate_results[-1]["width_img"] = width_img

        return grasps


class Img2WorldConverter:
    def __init__(
        self,
        coord_converter: CT.Img2WorldCoordConverter,
        decropper: Union[CT.DeCenterCrop, CT.DeCenterCropResized()] = None,
        height_adjuster: CT.GraspHeightAdjuster = None,
        intermediate_results_queue_size: int = 1,
    ) -> NDArray[Shape["3"], Float]:
        super().__init__()

        # sub converter
        self.coord_converter = coord_converter
        self.decropper = decropper or (lambda x, y: x)
        self.height_adjuster = height_adjuster or (lambda x: x)

        self.intermediate_results = deque(maxlen=intermediate_results_queue_size)

    def _decrop_grasp(
        self, grasp: ImageGrasp, orig_img_size: Tuple[int, int]
    ) -> ImageGrasp:
        # first we account for the fact that the image was resized and/or cropped
        center_decropped = self.decropper(grasp.center, orig_img_size)

        p1, p2 = get_antipodal_points(grasp.center, -grasp.angle, grasp.width)
        width_decropped = np.linalg.norm(
            self.decropper(p1, orig_img_size) - self.decropper(p2, orig_img_size)
        )
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

    def _get_antipodal_points_world(
        self,
        grasp_decropped: ImageGrasp,
        center_depth: float,
        cam_intrinsics: NDArray[Shape["3,3"], Float],
        cam_rot: NDArray[Shape["3,3"], Float],
        cam_pos: NDArray[Shape["3"], Float],
    ):
        antipodal_points_img = get_antipodal_points(
            grasp_decropped.center, -grasp_decropped.angle, grasp_decropped.width
        )

        antipodal_points_world = [
            self.coord_converter(p, center_depth, cam_intrinsics, cam_rot, cam_pos)
            for p in antipodal_points_img
        ]

        return antipodal_points_world

    def _get_width_world(
        self, antipodal_points_world: List[NDArray[Shape["3"], Float]]
    ):
        width_world = np.linalg.norm(
            antipodal_points_world[0] - antipodal_points_world[1]
        )

        return width_world

    def _get_angle_world(
        self,
        antipodal_points_world: List[NDArray[Shape["3"], Float]],
    ):
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
        cam_intrinsics: NDArray[Shape["3,3"], Float],
        cam_rot: NDArray[Shape["3,3"], Float],
        cam_pos: NDArray[Shape["3"], Float],
    ) -> RealGrasp:
        grasp_decropped = self._decrop_grasp(grasp, list(orig_depth_image.shape[1:]))

        # convert the grasp center to world frame
        center_depth = self._get_center_depth(grasp_decropped, orig_depth_image)
        center_world = self.coord_converter(
            grasp_decropped.center, center_depth, cam_intrinsics, cam_rot, cam_pos
        )

        antipodal_points_world = self._get_antipodal_points_world(
            grasp_decropped, center_depth, cam_intrinsics, cam_rot, cam_pos
        )

        # convert the width of the grasp
        width_world = self._get_width_world(antipodal_points_world)

        # convert the angle
        angle_world = self._get_angle_world(antipodal_points_world)

        grasp_world = RealGrasp(center_world, grasp.quality, angle_world, width_world)

        self.intermediate_results.append({})
        self.intermediate_results[-1]["grasp_decropped"] = grasp_decropped
        self.intermediate_results[-1]["center_depth"] = center_depth
        self.intermediate_results[-1]["grasp_world_raw"] = grasp_world

        grasp_world = self.height_adjuster(grasp_world)

        return grasp_world
