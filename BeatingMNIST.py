import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from scipy.ndimage import zoom

## Random
## Z
## Q - 4ch 
## 0 - LVOT 
## 1 - 3VV 
## I - 3VT 
## Random 


def get_mnist_digit(digit=3):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    images = mnist['data'].reshape(-1, 28, 28)
    labels = mnist['target'].astype(int)

    digit_images = images[labels == digit]
    idx = np.random.randint(len(digit_images))
    return digit_images[idx]


def simulate_heartbeat_sequence(img, num_frames=20, min_scale=0.9, max_scale=1.2):
    """
    Simulate a pulsing image by scaling it up and down to mimic a heartbeat.
    Handles both zoom in and zoom out cases safely.
    """
    h, w = img.shape
    center = np.array([h, w]) // 2
    frames = []

    # Generate scale factors (sinusoidal pulse)
    scales = (np.sin(np.linspace(0, 2 * np.pi, num_frames)) + 1) / 2
    scales = min_scale + (max_scale - min_scale) * scales

    for scale in scales:
        zoomed = zoom(img, scale, order=1)
        zh, zw = zoomed.shape
        canvas = np.zeros((h, w), dtype=zoomed.dtype)

        # Compute placement or cropping
        if zh <= h and zw <= w:
            # Center and pad
            top = (h - zh) // 2
            left = (w - zw) // 2
            canvas[top:top + zh, left:left + zw] = zoomed
        else:
            # Center crop
            top = (zh - h) // 2
            left = (zw - w) // 2
            zoomed_cropped = zoomed[top:top + h, left:left + w]
            canvas = zoomed_cropped

        frames.append(canvas)

    return np.array(frames)


# Example usage
digit_img = get_mnist_digit(3)
heartbeat_seq = simulate_heartbeat_sequence(digit_img, num_frames=60)

# Visualize the sequence
for i, frame in enumerate(heartbeat_seq):
    plt.imshow(frame, cmap='gray')
    plt.title(f"Frame {i}")
    plt.axis('off')
    plt.pause(0.02)
plt.close()
