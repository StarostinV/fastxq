import numpy as np

from fastxq import QInterpolation, PolarInterpolation


def test_q_conversion(raw_image, q_image):
    q_img, q_config = q_image
    q_interpolation = QInterpolation(**q_config)
    interpolated = q_interpolation(raw_image)
    assert np.allclose(q_img, interpolated)


def test_polar_conversion(raw_image, polar_image):
    polar_img, polar_config = polar_image
    polar_interpolation = PolarInterpolation(**polar_config)
    interpolated = polar_interpolation(raw_image)
    assert np.allclose(polar_img, interpolated)
