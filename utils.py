'''
Utilities functions for generating 
'''
import numpy as np
import cv2
from scipy.ndimage import affine_transform
from morphInterp import morph_interp

def gen_affine_matrix(
        scale_x: float, 
        scale_y: float,
        theta_deg: float,
        shift_x: float,
        shift_y: float,
        in_shape_h: int,
        in_shape_w: int, 
        out_shape_h: int,
        out_shape_w: int
        ) -> tuple:
    '''
    Given a list of parameters, 
    and generate a single affine matrix for transforming a single image.
    
    :param scale_x: scale factor in x direction
    :type scale_x: float
    :param scale_y: scale factor in y direction
    :type scale_y: float
    :param theta_deg: rotation angle in degrees, in degrees
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


def add_noise(
        frames: list[np.ndarray],
        gaussian_sigma: float = 0.50,
        sp_prob: float = 0.15,
        brightness_delta: float = 0.20,
        contrast_low: float = 0.80,
        contrast_high: float = 1.20,
        seed: int | None = None
    ) -> list[np.ndarray]:
    '''
    Add noise to a list of frames.
    The list of frames should from a single video.

    Combine Gaussian noise, salt–pepper noise, contrast jitter and
    brightness jitter into *fresh copies* of the input frames.

    :param frames: list of frames
    :type frames: list[np.ndarray]
    :param gaussian_sigma: sigma of Gaussian noise
    :type gaussian_sigma: float
    :param sp_prob: probability of salt–pepper noise
    :type sp_prob: float
    :param brightness_delta: delta of brightness jitter
    :type brightness_delta: float
    :param contrast_low: low bound of contrast jitter
    :type contrast_low: float
    :param contrast_high: high bound of contrast jitter
    :type contrast_high: float
    :param seed: seed for random number generator
    :type seed: int | None
    :return: list of frames with noise added
    '''
    rng  = np.random.default_rng(seed)
    out  = []

    # Pre-compute booleans so the `if` tests are cheap inside the loop
    use_gauss   = gaussian_sigma   > 0
    use_sp      = sp_prob          > 0
    use_bright  = brightness_delta > 0
    use_contrast= (contrast_low != 1.0) or (contrast_high != 1.0)

    for img in frames:
        f = img.astype(np.float32, copy=True)

        # ------------------------------------------------------------------ #
        # 1.  Gaussian noise
        # ------------------------------------------------------------------ #
        if use_gauss:
            f += rng.normal(0.0, gaussian_sigma, f.shape)

        # ------------------------------------------------------------------ #
        # 2.  Salt & pepper
        # ------------------------------------------------------------------ #
        if use_sp:
            m = rng.random(f.shape)
            f[m < 0.5 * sp_prob]       = 0.0   # pepper
            f[m > 1.0 - 0.5 * sp_prob] = 1.0   # salt

        # ------------------------------------------------------------------ #
        # 3.  Contrast
        # ------------------------------------------------------------------ #
        if use_contrast:
            c = rng.uniform(contrast_low, contrast_high)
            f = (f - 0.5) * c + 0.5

        # ------------------------------------------------------------------ #
        # 4.  Brightness
        # ------------------------------------------------------------------ #
        if use_bright:
            b = rng.uniform(-brightness_delta, brightness_delta)
            f += b

        # ------------------------------------------------------------------ #
        # 5.  Clip back to valid range
        # ------------------------------------------------------------------ #
        out.append(np.clip(f, 0.0, 1.0))

    return out


def floatImage2uint8(image: np.ndarray) -> np.ndarray:
    '''
    Convert a float image to uint8 image.
    '''
    return (image * 255).astype(np.uint8)


def save_video_opencv(video_frames: list[np.ndarray], save_path: str, fps: int) -> None:
    '''
    Save a list of video frames to an MP4 file using OpenCV.
    
    :param video_frames: list of video frames as numpy arrays
    :type video_frames: list[np.ndarray]
    :param save_path: path to save the video file
    :type save_path: str
    :param fps: frames per second for the output video
    :type fps: int
    :return: None
    :rtype: None
    '''
    if not video_frames:
        raise ValueError("video_frames list is empty")
    
    # Get video dimensions from the first frame
    height, width = video_frames[0].shape[:2]
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height), isColor=False)
    
    # Write each frame to the video
    for frame in video_frames:
        # Ensure frame is in the correct format (uint8)
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        out.write(frame)
    
    # Release the VideoWriter
    out.release()