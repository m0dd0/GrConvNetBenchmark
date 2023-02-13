import numpy as np
from nptyping import NDArray, Float, Shape


def get_antipodal_points(
    center: NDArray[Shape["2"], Float], angle: float, width: float
) -> NDArray[Shape["2, 2"], Float]:
    """Get the two antipodal points of a grasp.
    The angle of the grasp is given in radians and is measured wrt to a horizontal
    x-axis pointing to the right and vertical y-axis pointing to the top.
    If you are in image space you need to account that the image coordinate system
    is flipped in y direction compared to the coordinate system the angle is measured in.
    So in order to get the correct antipodal points in image space you need to
    change the sign of the angle passed to this function.

    Args:
        center: _description_
        angle: _description_
        width: _description_

    Returns:
        _type_: _description_
    """
    delta = np.array([np.cos(angle), np.sin(angle)]) * width / 2
    return np.array([center - delta, center + delta])
