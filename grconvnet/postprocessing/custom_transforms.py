"""_summary_
"""

from typing import Tuple
from copy import deepcopy


from nptyping import NDArray, Shape, Float, Int
import numpy as np
from skimage.filters import gaussian
from skimage.feature import peak_local_max

from grconvnet.datatypes import RealGrasp


class AngleConverter:
    def __call__(
        self,
        sin_cos_img: Tuple[
            NDArray[Shape["H, W"], Float], NDArray[Shape["H, W"], Float]
        ],
    ) -> NDArray[Shape["H, W"], Float]:
        sin_img, cos_img = sin_cos_img
        # return torch.atan2(sin_img, cos_img) / 2.0
        return np.arctan2(sin_img, cos_img) / 2.0


class SkGaussian:
    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(
        self, img: NDArray[Shape["H, W"], Float]
    ) -> NDArray[Shape["H, W"], Float]:
        return gaussian(img, self.sigma)


class GraspLocalizer:
    def __init__(
        self, min_distance: int = 20, threshold: float = 0.2, n_grasps: int = 1
    ):
        self.min_distance = min_distance
        self.threshold = threshold
        self.n_grasps = n_grasps

    def __call__(
        self, q_img: NDArray[Shape["H, W"], Float]
    ) -> NDArray[Shape["N_grasps, 2"], Int]:
        return peak_local_max(
            q_img,
            min_distance=self.min_distance,
            threshold_abs=self.threshold,
            num_peaks=self.n_grasps,
            exclude_border=False,
        )


class Scaler:
    def __init__(self, factor: int):
        self.factor = factor

    def __call__(
        self, img: NDArray[Shape["H, W"], Float]
    ) -> NDArray[Shape["H, W"], Float]:
        return img * self.factor


class GraspHeightAdjuster:
    def __init__(self, min_height: float = 0.01, target_grasp_depth: float = 0.04):
        self.min_height = min_height
        self.target_grasp_depth = target_grasp_depth

    def __call__(self, grasp: RealGrasp) -> RealGrasp:
        adjusted_grasp = deepcopy(grasp)
        grasp_height = max(self.min_height, grasp.center[2] - self.target_grasp_depth)
        adjusted_grasp.center[2] = grasp_height

        return adjusted_grasp


class LegacyGraspHeightAdjuster:
    def __init__(self, z_offset: float = -0.04):
        self.z_offset = z_offset

    def __call__(self, grasp: RealGrasp) -> RealGrasp:
        adjusted_grasp_center = grasp.center.copy()
        adjusted_grasp_center[2] = self.z_offset

        adjusted_grasp = RealGrasp(
            adjusted_grasp_center, grasp.quality, grasp.angle, grasp.width
        )

        return adjusted_grasp


class DeCenterCrop:
    def __call__(
        self,
        coordinates: NDArray[Shape["2"], Float],
        original_img_size: Tuple[int, int],
    ) -> NDArray[Shape["2"], Float]:
        coord_offset = np.array(original_img_size) / 2 - 112
        coord_offset = coord_offset[::-1]
        scaling_factor = 1

        coordinates = coordinates * scaling_factor + coord_offset

        return coordinates


class DeCenterCropResized:
    def __call__(
        self,
        coordinates: NDArray[Shape["2"], Float],
        original_img_size: Tuple[int, int],
    ) -> NDArray[Shape["2"], Float]:
        delta = (max(original_img_size) - min(original_img_size)) / 2
        coord_offset = np.zeros(2)

        if original_img_size[0] > original_img_size[1]:
            # landscape format
            coord_offset[1] = delta
        else:
            # portrait format
            coord_offset[0] = delta

        scaling_factor = min(original_img_size) / 224

        coordinates = coordinates * scaling_factor + coord_offset

        return coordinates  # (x,y)


# class Decropper:
#     def __init__(self, resized_in_preprocess: bool, original_img_size: Tuple[int, int]):
#         """_summary_

#         Args:
#             resized_in_preprocess (bool): _description_
#             original_img_size (Tuple[int, int]): Image size in (width, height)
#         """
#         self.resized_in_preprocess = resized_in_preprocess
#         self.original_img_size = original_img_size

#     def __call__(
#         self, coordinates: NDArray[Shape["2"], Float]
#     ) -> NDArray[Shape["2"], Float]:
#         if not self.resized_in_preprocess:
#             coord_offset = np.array(self.original_img_size) / 2 - 112
#             coord_offset = coord_offset[::-1]
#             scaling_factor = 1

#         else:
#             delta = (max(self.original_img_size) - min(self.original_img_size)) / 2
#             coord_offset = np.zeros(2)
#             if self.original_img_size[0] > self.original_img_size[1]:
#                 # landscape format
#                 coord_offset[1] = delta
#             else:
#                 # portrait format
#                 coord_offset[0] = delta
#             scaling_factor = min(self.original_img_size) / 224

#         coordinates = coordinates * scaling_factor + coord_offset

#         return coordinates  # (x,y)


class World2ImgCoordConverter:
    def __call__(
        self,
        p_world: NDArray[Shape["3"], Float],
        cam_intrinsics: NDArray[Shape["3,3"], Float],
        cam_rot: NDArray[Shape["3,3"], Float],
        cam_pos: NDArray[Shape["3"], Float],
    ) -> NDArray[Shape["2"], Float]:
        # p_cam = R @ (p_world - T)
        # p_img_h = K @ p_cam = [[p_ix*p_cz]
        #                        [p_iy*p_cz]
        #                        [p_cz     ]]
        # p_img = [[p_ix]  = (p_img_h / p_cz)[:2] = (p_img_h / p_img_h[2])[:2]
        #           p_iy]]

        p_world = p_world.reshape((3, 1))  # (3,1)
        p_cam = cam_rot @ (p_world - cam_pos)
        p_img_h = cam_intrinsics @ p_cam
        p_img = (p_img_h / p_img_h[2])[:2].flatten()  # (2,)

        return p_img


class Img2WorldCoordConverter:
    def __call__(
        self,
        p_img: NDArray[Shape["2"], Int],
        p_cam_z: float,
        cam_intrinsics: NDArray[Shape["3,3"], Float],
        cam_rot: NDArray[Shape["3,3"], Float],
        cam_pos: NDArray[Shape["3"], Float],
    ) -> NDArray[Shape["3"], Float]:
        # K = [[fx 0  cx]
        #      [0  fy cy]
        #      [0  0  1 ]]

        # p_cam = R @ (p_world - T) = [[p_cx] <--> p_world = (inv(R) @ p_cam) + T
        #                             [p_cy]
        #                             [p_cz]]
        # p_img_h = K @ p_cam = [[fx*p_cx + cx*p_cz]  = [[fx*p_cx/p_cz + cx] * p_cz
        #                        [fy*p_cy + cy*p_cz]     [fy*p_cy/p_cz + cy]
        #                        [p_cz             ]]    [1             ]]
        # p_img = [[p_ix]  = [[fx*p_cx/p_cz + cx]   <--> p_cam = [[p_cx]  = [[(p_ix - cx)*p_cz/fx]
        #          [p_iy]]    [fy*p_cy/p_cz + cy]]                [p_cy]     [(p_iy - cy)*p_cz/fy]
        #                                                         [p_cz]]    [p_cz]]
        cam_pos = cam_pos.reshape((3, 1))
        cam_rot_inv = np.linalg.inv(cam_rot)

        p_img = p_img.reshape(2, 1)
        cx = cam_intrinsics[0, 2]
        cy = cam_intrinsics[1, 2]
        fx = cam_intrinsics[0, 0]
        fy = cam_intrinsics[1, 1]
        p_img_x = p_img[0, 0]
        p_img_y = p_img[1, 0]

        p_cam_x = (p_img_x - cx) * p_cam_z / fx
        p_cam_y = (p_img_y - cy) * p_cam_z / fy
        p_cam = np.array([p_cam_x, p_cam_y, p_cam_z]).reshape((3, 1))

        p_world = cam_rot_inv @ p_cam + cam_pos

        p_world = p_world.flatten()

        return p_world
