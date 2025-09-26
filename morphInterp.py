import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from scipy.ndimage import distance_transform_edt, gaussian_filter



def signed_dt(mask: np.ndarray) -> np.ndarray:
    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(~mask)
    return inside - outside         # positive inside, negative outside


def morph_interp(start_image, end_image, n_frames=20):
    bin0 = start_image > 0.5
    bin1 = end_image > 0.5
    D0 = signed_dt(bin0)
    D1 = signed_dt(bin1)
    interped_frames = []
    for t in np.linspace(0, 1, n_frames):
        Dt = (1 - t) * D0 + t * D1
        mask_t = Dt >= 0
        frame = gaussian_filter(mask_t.astype(float), sigma=0.7)
        interped_frames.append(frame)
    return interped_frames


# -----------------------------------------------------------------
# 1. Fetch two sample digits (0 and 1) and normalise to [0,1]
# ------------------------------------------------------------------
def get_digit(digit: int):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.reshape(-1, 28, 28) / 255.0
    y = mnist.target.astype(int)
    idx = np.where(y == digit)[0][0]
    return X[idx]

if __name__ == "__main__":
    img0 = get_digit(0)
    img1 = get_digit(1)
    frames = morph_interp(img0, img1, 200)

# ------------------------------------------------------------------
# 6. Quick viewer
# ------------------------------------------------------------------
    plt.figure(figsize=(3, 3))
    for i, f in enumerate(frames):
        plt.clf()
        plt.imshow(f, cmap="gray")
        plt.title(f"t = {i/(200-1):.2f}")
        plt.axis('off')
        plt.pause(0.05)
    plt.close()
