"""_summary_
"""

from typing import List

from matplotlib import pyplot as plt
import numpy as np

from grconvnet.datatypes import ImageGrasp, RealGrasp
from grconvnet.utils.geometry import get_antipodal_points
from grconvnet.postprocessing import World2ImgCoordConverter


def make_tensor_displayable(
    tensor, convert_chw: bool = False, convert_to_int: bool = False
):
    tensor = np.array(tensor)
    tensor = np.squeeze(tensor)  # convert one channel images to 2d tensors
    assert len(tensor.shape) in [2, 3], "squeezed Tensor must be 2 or 3 dimensional"

    if convert_chw:
        assert tensor.shape[0] in [1, 3], "first dimension must be 1 or 3"
        # chw -> hwc
        tensor = np.transpose(tensor, (1, 2, 0))

    if convert_to_int:
        tensor = tensor.astype("uint8")

    return tensor


def image_grasps_ax(
    ax, background, image_grasps: List[ImageGrasp], annotate: bool = True
):
    ax.imshow(background)

    for grasp in image_grasps:
        ax.scatter(x=grasp.center[0], y=grasp.center[1])

        antipodal_points = get_antipodal_points(
            grasp.center[0:2], -grasp.angle, grasp.width
        )

        ax.plot(antipodal_points[:, 0], antipodal_points[:, 1])

        if annotate:
            ax.annotate(
                f"c:{(int(grasp.center[0]), int(grasp.center[1]))}\n"
                + f"q: {round(grasp.quality, 3)}\n"
                + f"a: {round(np.rad2deg(grasp.angle), 3)}\n"
                + f"w: {round(grasp.width, 3)}",
                xy=grasp.center[0:2],
            )


def world_grasps_ax(
    ax,
    backgound,
    grasps: List[RealGrasp],
    cam_intrinsics,
    cam_rot,
    cam_pos,
    annotate: bool = True,
):
    ax.imshow(backgound)

    world2img_converter = World2ImgCoordConverter(cam_intrinsics, cam_rot, cam_pos)

    for grasp in grasps:
        center_img = world2img_converter(grasp.center)
        ax.scatter(x=center_img[0], y=center_img[1])

        antipodal_points_world = get_antipodal_points(
            grasp.center[0:2], grasp.angle, grasp.width
        )
        antipodal_points_world = np.hstack(
            (antipodal_points_world, np.full((2, 1), grasp.center[2]))
        )

        antipodal_points_img = np.array(
            [world2img_converter(p) for p in antipodal_points_world]
        )
        ax.plot(antipodal_points_img[:, 0], antipodal_points_img[:, 1])

        if annotate:
            ax.annotate(
                f"c: {tuple(grasp.center.round(3))}\n"
                + f"q: {round(grasp.quality, 3)}\n"
                + f"a: {round(np.rad2deg(grasp.angle), 3)}\n"
                + f"w: {round(grasp.width, 3)}",
                xy=center_img[0:2],
            )


def overview_fig(
    original_rgb=None,
    preprocessed_rgb=None,
    q_img=None,
    angle_img=None,
    width_img=None,
    image_grasps=None,
    world_grasps=None,
    cam_intrinsics=None,
    cam_rot=None,
    cam_pos=None,
    fig=None,
):
    assert (
        cam_intrinsics is not None and cam_rot is not None and cam_pos is not None
        if world_grasps
        else True
    )

    if fig is None:
        fig = plt.figure()

    if original_rgb is not None:
        ax = fig.add_subplot(3, 3, 1)
        ax.imshow(original_rgb)
        ax.set_title("Original Image")

    if preprocessed_rgb is not None:
        ax = fig.add_subplot(3, 3, 2)
        ax.imshow(preprocessed_rgb)
        ax.set_title("Preprocessed Image")

    if q_img is not None:
        ax = fig.add_subplot(3, 3, 4)
        ax.imshow(q_img)
        ax.set_title("Quality Image")

    if angle_img is not None:
        ax = fig.add_subplot(3, 3, 5)
        ax.imshow(angle_img)
        ax.set_title("Angle Image")

    if width_img is not None:
        ax = fig.add_subplot(3, 3, 6)
        ax.imshow(width_img)
        ax.set_title("Width Image")

    if image_grasps is not None:
        ax = fig.add_subplot(3, 3, 7)
        image_grasps_ax(ax, preprocessed_rgb, image_grasps)
        ax.set_title("Image Grasps")

    if world_grasps is not None:
        ax = fig.add_subplot(3, 3, 8)
        world_grasps_ax(
            ax,
            original_rgb,
            world_grasps,
            cam_intrinsics,
            cam_rot,
            cam_pos,
        )
        ax.set_title("World Grasps")

    return fig


# def show_closed_figure(fig):
#     dummy = plt.figure()
#     new_manager = dummy.canvas.manager
#     new_manager.canvas.figure = fig
#     fig.set_canvas(new_manager.canvas)

#     plt.show()
