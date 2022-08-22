from pathlib import Path

from PIL import Image
import numpy as np

import pytest

PATH_TO_RAW_IMAGE: Path = Path(__file__).parent / 'data' / 'raw' / 'img1.tif'
PATH_TO_Q_IMAGE: Path = Path(__file__).parent / 'data' / 'processed' / 'q_img.npy'
PATH_TO_POLAR_IMAGE: Path = Path(__file__).parent / 'data' / 'processed' / 'polar_img.npy'

Q_IMG_CONFIG: dict = {
    'distance': 809,
    'flip_y': False,
    'flip_z': True,
    'incidence_angle': 0.5,
    'pixel_size': 0.2,
    'q_xy_max': 2.7,
    'q_xy_size': 1350,
    'q_z_max': 2.7,
    'q_z_size': 1350,
    'wavelength': 0.6888,
    'y0': 545,
    'z0': 222,
}
POLAR_IMG_CONFIG: dict = {
    'distance': 809,
    'flip_y': False,
    'flip_z': True,
    'incidence_angle': 0.5,
    'pixel_size': 0.2,
    'q_xy_max': 2.7,
    'polar_q_size': 1024,
    'q_z_max': 2.7,
    'polar_angular_size': 512,
    'wavelength': 0.6888,
    'y0': 545,
    'z0': 222,
}


@pytest.fixture
def raw_image():
    img = np.array(Image.open(PATH_TO_RAW_IMAGE))
    return img


@pytest.fixture
def q_image():
    q_img = np.load(PATH_TO_Q_IMAGE)
    q_config = dict(Q_IMG_CONFIG)
    return q_img, q_config


@pytest.fixture
def polar_image():
    polar_img = np.load(PATH_TO_POLAR_IMAGE)
    polar_config = dict(POLAR_IMG_CONFIG)
    return polar_img, polar_config
