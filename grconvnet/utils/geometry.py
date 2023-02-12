import numpy as np
from nptyping import NDArray, Float, Shape


def get_antipodal_points(
    center: NDArray[Shape["2"], Float], angle: float, width: float
) -> NDArray[Shape["2, 2"], Float]:
    delta = np.array([np.cos(angle), np.sin(angle)]) * width / 2
    # because the image coordinate system is flipped in y direction compared to the world coordinate system we have to flip the delta vector in y direction
    delta[1] = -delta[1]
    return np.array([center - delta, center + delta])
