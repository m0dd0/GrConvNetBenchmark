import numpy as np
from nptyping import NDArray, Float, Shape


def get_antipodal_points(
    center: NDArray[Shape["2"], Float],
    angle: float,
    width: float,
) -> NDArray[Shape["2, 2"], Float]:
    """Get the two antipodal points of a grasp.
    The angle of is assumed to be given in radians and is measured wrt to the x-axis
    in counter clockwise direction.
    If you are in image space you need to account that the image coordinate system
    is flipped in y direction compared to the coordinate which is reference for the
    angle outputted by the network. Therefore you need to multiply the angle by -1
    to obtain correct results in the image space.

    Args:
        center: _description_
        angle: _description_
        width: _description_

    Returns:
        _type_: _description_
    """
    # if flipped_y:
    #     angle = -angle
    # angle = angle - system_angle

    delta = np.array([np.cos(angle), np.sin(angle)]) * width / 2
    antipodal_points = np.array([center - delta, center + delta])

    return antipodal_points
