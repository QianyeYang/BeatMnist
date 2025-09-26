import numpy as np
from scipy.ndimage import affine_transform
from beatMnist_utils import load_emnist
from matplotlib import pyplot as plt
from scipy.ndimage import affine_transform, gaussian_filter1d
import imageio
import utils


def smooth_random_offsets(
        max_px, 
        time_points,
        cutoff_hz=0.4,
        seed=None,
        ) -> tuple:
    """
    Two independent band-limited noise sequences (dx, dy) in pixels.
    cutoff_hz ≈ how quickly the drift can change direction (lower = calmer).
    """
    rng = np.random.default_rng(seed)
    t   = time_points
    n   = len(t)

    sigma = 60 / (2 * np.pi * cutoff_hz)   # heuristic
    x_raw = gaussian_filter1d(rng.standard_normal(n), sigma=sigma)
    y_raw = gaussian_filter1d(rng.standard_normal(n), sigma=sigma)

    # normalise & scale
    x_shift = max_px * x_raw / np.max(np.abs(x_raw))
    y_shift = max_px * y_raw / np.max(np.abs(y_raw))
    return x_shift, y_shift


def periodic_random_scale_factor(
        mu=1.5,
        amp=0.5,
        delta=0.15,          # ±5 % jitter
        f=2.0,               # Hz
        time_points=None,
        seed=None,
        ) -> tuple:
    """
    Return time vector t and signal y whose peaks/valleys vary
    up to ±delta·amp but whose period is fixed at 1/f.
    """
    rng  = np.random.default_rng(seed)
    t    = time_points
    cyc  = np.floor(t * f).astype(int)      # integer cycle index
    n_cyc = cyc.max() + 1                   # total cycles

    # draw random scale factors for positive & negative half-cycles
    a_pos = rng.uniform(1-delta, 1+delta, size=n_cyc)
    a_neg = rng.uniform(1-delta, 1+delta, size=n_cyc)

    # envelope: choose factor according to sign of sine
    half = (t * f) % 1.0 < 0.5              # True = first (positive) half
    envelope = np.where(half, a_pos[cyc], a_neg[cyc]) * amp

    y = mu + envelope * np.sin(2 * np.pi * f * t)
    return y


def periodic_random_amplitude_rotation(
    theta_left,
    theta_right,
    f_rot,
    time_points,
    ) -> np.ndarray:
    '''
    Generate a sequence of rotation, each element represent a rotation angle in degrees.

    :param theta_left: leftmost rotation angle in degrees
    :type theta_left: float
    :param theta_right: rightmost rotation angle in degrees
    :type theta_right: float
    :param f_rot: rotation frequency in Hz
    :type f_rot: float
    :param time_points: time points
    :type time_points: np.ndarray

    :return: sequence of rotation angles in degrees
    :rtype: np.ndarray
    '''
    amp_theta  = 0.5 * (theta_right - theta_left)
    centre_th  = 0.5 * (theta_right + theta_left)
    theta_t    = centre_th + amp_theta * np.sin(2 * np.pi * f_rot * time_points)
    return theta_t


def affine_matrix_single_point(
        scale_x, 
        scale_y,
        theta_deg,
        shift_x,
        shift_y,
        in_shape_h,
        in_shape_w, 
        out_shape_h,
        out_shape_w
        ) -> tuple:
    '''
    Generate an affine matrix and corresponding offset based on the given parameters.

    :param scale_x: scale factor in x direction
    :type scale_x: float
    :param scale_y: scale factor in y direction
    :type scale_y: float
    :param theta_deg: rotation angle in degrees
    :type theta_deg: float
    :param shift_x: shift in x direction, in pixels
    :type shift_x: float
    :param shift_y: shift in y direction, in pixels
    :type shift_y: float
    :param in_shape_h: height of the input image
    :type in_shape_h: int
    :param in_shape_w: width of the input image
    :type in_shape_w: int
    :param out_shape_h: height of the output image
    :type out_shape_h: int
    :param out_shape_w: width of the output image
    :type out_shape_w: int

    :return: a tuple of (affine_matrix, offset)
    :rtype: tuple
    '''

    theta = np.deg2rad(theta_deg)
    R = np.array([[ np.cos(theta), -np.sin(theta)],
                  [ np.sin(theta),  np.cos(theta)]])
    S = np.diag([scale_x, scale_y])
    A  = R @ S                     # scale → rotate
    in_c  = np.array([(in_shape_w - 1) / 2.0, (in_shape_h - 1) / 2.0])
    out_c = np.array([(out_shape_w - 1) / 2.0, (out_shape_h - 1) / 2.0])

    t      = out_c + np.array([shift_x, shift_y]) - A @ in_c      # translation
    A_inv  = np.linalg.inv(A)
    offset = -A_inv @ t

    return A_inv, offset


class AffineTransformer:
    def __init__(
            self,
            rotation_theta_left: float,
            rotation_theta_right: float,
            rotation_f_rot: float,
            offset_max_px: float,
            offset_cutoff_hz: float,
            scale_mu: float,
            scale_amp: float,
            scale_delta: float,
            scale_f: float,  # scale frequency in Hz
            seed: int = None,
    ):
        self.__rotation_theta_left = rotation_theta_left
        self.__rotation_theta_right = rotation_theta_right
        self.__rotation_f_rot = rotation_f_rot
        self.__offset_max_px = offset_max_px
        self.__offset_cutoff_hz = offset_cutoff_hz
        self.__scale_mu = scale_mu
        self.__scale_amp = scale_amp
        self.__scale_delta = scale_delta
        self.__scale_f = scale_f
        self.__seed = seed

    def transform(
            self,
            list_of_images: list,
            out_shape: tuple,
            order: int = 1,
            fps: int = 60,
    ):
        in_shapes = [im.shape for im in list_of_images]
        assert len(set(in_shapes)) == 1, "All images must have the same shape!"
        in_shape = in_shapes[0]
        in_h, in_w = in_shape
        out_h, out_w = out_shape

        video_length = len(list_of_images)
        time_points = np.array(range(video_length)) * (1/fps)

        y = periodic_random_scale_factor(
            mu = self.__scale_mu,
            amp = self.__scale_amp,
            delta = self.__scale_delta,
            f = self.__scale_f,
            time_points=time_points,
            seed=self.__seed
            )
        
        theta_t = periodic_random_amplitude_rotation(
            theta_left = self.__rotation_theta_left,
            theta_right = self.__rotation_theta_right,
            f_rot = self.__rotation_f_rot,
            time_points = time_points
        )

        dx_t, dy_t = smooth_random_offsets(
            max_px = self.__offset_max_px,
            time_points=time_points,
            cutoff_hz = self.__offset_cutoff_hz,
            seed=self.__seed
        )

        affine_mats = []
        affine_offsets = []
        for s, th, dx, dy in zip(y, theta_t, dx_t, dy_t):
            A_inv, off = affine_matrix_single_point(s, s, th, dx, dy, in_h, in_w, out_h, out_w)
            affine_mats.append(A_inv)
            affine_offsets.append(off)
        affine_mats = np.stack(affine_mats)
        affine_offsets = np.stack(affine_offsets)
        frames = []

        for idx, (A_inv, off) in enumerate(zip(affine_mats, affine_offsets)):
            frames.append(
                affine_transform(
                    list_of_images[idx],
                    matrix=A_inv,
                    offset=off,
                    output_shape=(out_h, out_w),
                    order=order,
                    mode='constant',
                    cval=0.0
                )
            )
        return frames



if __name__ == "__main__":

    im_sample = load_emnist()[0][0]
    new_h, new_w = 112, 112
    order = 1
    h, w = 28, 28

    aft = AffineTransformer(
        rotation_theta_left = -30,
        rotation_theta_right = 30,
        rotation_f_rot = 0.5,  # rotation frequency in Hz
        offset_max_px = 40,   # maximum offset in pixels
        offset_cutoff_hz = 0.4,
        scale_mu = 1.5,
        scale_amp = 0.5,
        scale_delta = 0.15,
        scale_f = 2.0,
    )
    frames = aft.transform(
        list_of_images = [im_sample] * 400,
        out_shape = (new_h, new_w),
        order = order,
        fps = 60
    )
    imageio.mimsave('new_affine.mp4', frames, fps=60)
