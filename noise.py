import numpy as np


def add_noise2frames(
        frames,
        gaussian_sigma   = 0.50,     # σ of additive N(0,σ²) noise
        sp_prob          = 0.15,     # salt-&-pepper pixel probability
        brightness_delta = 0.20,     # ±fractional shift
        contrast_low     = 0.80,     # contrast factor drawn ∈ [low, high]
        contrast_high    = 1.20,
        seed             = None,     # RNG seed for reproducibility
    ):
    """
    Combine Gaussian noise, salt–pepper noise, contrast jitter and
    brightness jitter into *fresh copies* of the input frames.

    Parameters
    ----------
    frames : list[np.ndarray]
        Each frame must be float32/64 in [0,1].
    gaussian_sigma : float ≥ 0
        σ of additive zero-mean Gaussian noise.  0 → disabled.
    sp_prob : float in [0,1]
        Per-pixel probability of being set to 0 or 1 (50 / 50).  0 → off.
    brightness_delta : float ≥ 0
        Frame-wise brightness shift drawn from U(-Δ, +Δ).  0 → off.
    contrast_low, contrast_high : float
        Contrast factor drawn from U(low, high).  1.0 → no change.
    seed : int | None
        RNG seed for reproducibility.

    Returns
    -------
    list[np.ndarray]  (float32, clipped to [0,1])
    """
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


# frames = augment_frames(
#     frames,
#     gaussian_sigma=1.2,
#     sp_prob=0,
#     brightness_delta=0,
#     contrast_low=0.8,
#     contrast_high=1.2,
#     seed=123
#     )