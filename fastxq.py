import numpy as np
import cv2 as cv

__author__ = 'Vladimir Starostin'
__email__ = 'vladimir.starostin@uni-tuebingen.de'
__version__ = '0.0.1'

__all__ = [
    'QInterpolation',
    'PolarInterpolation',
    'convert_img',
    'get_detector_q_grid',
    'get_detector_polar_grid',
]


class QInterpolation(object):
    def __init__(self,
                 q_xy_max: float,
                 q_z_max: float,
                 q_xy_size: int,
                 q_z_size: int,
                 y0: float,
                 z0: float,
                 wavelength: float,
                 distance: float,
                 pixel_size: float,
                 algorithm: int = cv.INTER_LINEAR,
                 flip_y: bool = False,
                 flip_z: bool = False,
                 ):

        self._init_config(
            y0=y0,
            z0=z0,
            wavelength=wavelength,
            distance=distance,
            pixel_size=pixel_size,
            q_xy_max=q_xy_max,
            q_z_max=q_z_max,
            q_xy_size=q_xy_size,
            q_z_size=q_z_size,
        )

        self._flip_y, self._flip_z = flip_y, flip_z
        self._algorithm = algorithm

        self._xy, self._zz = self._get_grid()

        if self._algorithm not in (cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_LANCZOS4):
            self._algorithm = cv.INTER_LINEAR

    def _init_config(self, **kwargs):
        self._config = dict(kwargs)

    def _get_grid(self):
        return get_detector_q_grid(**self._config)

    def __call__(self, img: np.ndarray):
        img = self.flip(img)
        return convert_img(img, self._xy, self._zz, self._algorithm)

    def flip(self, img: np.ndarray):
        if self._flip_y:
            img = np.flip(img, 1)
        if self._flip_z:
            img = np.flip(img, 0)
        return img

    def __repr__(self):
        kwargs = ', '.join(f'{k}={str(v)}' for k, v in self._config.items())
        return f'{self.__class__.__name__}({kwargs})'


class PolarInterpolation(QInterpolation):
    def __init__(self,
                 q_xy_max: float,
                 q_z_max: float,
                 polar_q_size: int,
                 polar_angular_size: int,
                 y0: float,
                 z0: float,
                 wavelength: float,
                 distance: float,
                 pixel_size: float,
                 algorithm: int = cv.INTER_LINEAR,
                 flip_y: bool = False,
                 flip_z: bool = False,
                 ):
        super().__init__(
            q_xy_max, q_z_max, polar_q_size, polar_angular_size, y0, z0, wavelength, distance,
            pixel_size, algorithm, flip_y, flip_z,
        )

    def _init_config(self, **kwargs):
        kwargs['polar_q_size'] = kwargs.pop('q_xy_size')
        kwargs['polar_angular_size'] = kwargs.pop('q_z_size')

        self._config = dict(kwargs)

    def _get_grid(self):
        return get_detector_polar_grid(**self._config)


def convert_img(img: np.ndarray, xy: np.ndarray, zz: np.ndarray, algorithm: int = cv.INTER_LINEAR):
    return cv.remap(img.astype(np.float32), xy.astype(np.float32), zz.astype(np.float32), algorithm)


def get_detector_q_grid(
        q_xy_max: float,
        q_z_max: float,
        q_xy_size: int,
        q_z_size: int,
        y0: float,
        z0: float,
        wavelength: float,
        distance: float,
        pixel_size: float,

):
    q_xy, q_z = _get_q_grid(q_xy_max=q_xy_max, q_z_max=q_z_max, q_xy_size=q_xy_size, q_z_size=q_z_size)
    xy, zz = _get_detector_grid(
        q_xy=q_xy,
        q_z=q_z,
        y0=y0,
        z0=z0,
        wavelength=wavelength,
        distance=distance,
        pixel_size=pixel_size,
    )
    return xy, zz


def get_detector_polar_grid(
        q_xy_max: float,
        q_z_max: float,
        polar_q_size: int,
        polar_angular_size: int,
        y0: float,
        z0: float,
        wavelength: float,
        distance: float,
        pixel_size: float,
):
    q_xy, q_z = _get_q_polar_grid(q_xy_max, q_z_max, polar_q_size, polar_angular_size)

    xy, zz = _get_detector_grid(
        q_xy=q_xy,
        q_z=q_z,
        y0=y0,
        z0=z0,
        wavelength=wavelength,
        distance=distance,
        pixel_size=pixel_size,
    )
    return xy, zz


def _get_detector_grid(
        q_xy: np.ndarray,
        q_z: np.ndarray,
        y0: float,
        z0: float,
        wavelength: float,
        distance: float,
        pixel_size: float,
):
    k = 2 * np.pi / wavelength

    q_xy2, q_z2 = (q_xy / k) ** 2, (q_z / k) ** 2

    a = distance / pixel_size

    yn = 1 - (q_xy2 - 1) / (1 - q_z2)

    a2 = a ** 2

    yy2 = 4 * a2 / yn ** 2 / (1 - q_z2) - a2

    zz2 = (a2 + yy2) * q_z2 / (1 - q_z2)

    yy, zz = np.sqrt(yy2) + y0, np.sqrt(zz2) + z0

    return yy, zz


def _get_q_grid(q_xy_max: float, q_z_max: float, q_xy_size: int, q_z_size: int):
    q_xy = np.linspace(0, q_xy_max, q_xy_size)
    q_z = np.linspace(0, q_z_max, q_z_size)

    q_xy, q_z = np.meshgrid(q_xy, q_z)
    return q_xy, q_z


def _get_q_polar_grid(q_xy_max: float, q_z_max: float, polar_q_size: int, polar_angular_size: int):
    q_max = np.sqrt(q_xy_max ** 2 + q_z_max ** 2)

    r = np.linspace(0, q_max, polar_q_size)
    phi = np.linspace(0, np.pi / 2, polar_angular_size)

    rr, pp = np.meshgrid(r, phi)

    q_z = rr * np.sin(pp)
    q_xy = rr * np.cos(pp)

    return q_xy, q_z
