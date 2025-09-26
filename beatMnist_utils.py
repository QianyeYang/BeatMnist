"""
toy_emnist_video.py
-------------------
Create synthetic beating videos from EMNIST digits & letters.

Author: YOU
"""
from pathlib import Path
import random, math, os
import numpy as np
import torch
from torchvision.datasets import EMNIST
from scipy.ndimage import (
    zoom as ndi_zoom,
    distance_transform_edt,
    gaussian_filter,
)
from torch.utils.data import Dataset, DataLoader


def load_emnist(root="emnist_data"):
    ds = EMNIST(root=root, split="byclass", download=True)

    images = ds.data.numpy().astype(np.float32) / 255.0   # (N, 28, 28)
    labels = ds.targets.numpy()

    # --- mapping ---
    # Try to load the mapping file from disk
    mapping_path = Path(ds.root) / "EMNIST" / "raw" / "emnist-byclass-mapping.txt"
    if not mapping_path.exists():
        # Download the mapping file if not present
        import urllib.request
        url = "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"
        # The mapping file is inside the zip, but torchvision should have extracted it. If not, raise error.
        raise FileNotFoundError(f"Mapping file not found at {mapping_path}. Please ensure EMNIST is fully downloaded and extracted.")
    mapping_arr = np.loadtxt(mapping_path, dtype=int)
    idx_to_char = {int(lbl): chr(int(ascii_)) for lbl, ascii_ in mapping_arr}

    return images, labels, idx_to_char


emnist_images, emnist_labels, emnist_idx_to_char = load_emnist()
emnist_char_to_idx = {v: k for k, v in emnist_idx_to_char.items()}


def get_element(symbol):
    '''
    Provide symbol and get all the data from it.
    '''
    idx = np.where(emnist_labels == emnist_char_to_idx[symbol])
    return emnist_images[idx], emnist_labels[idx]

background_symbols = np.concatenate([get_element('Z')[0], get_element('z')[0]], axis=0)
fch_symbols = get_element('Q')[0]
lvot_symbols = get_element('0')[0]
three_vv_symbols = get_element('I')[0]
three_vt_symbols = get_element('L')[0]

p_train, p_val, p_test = 0.8, 0.1, 0.1

data_dict = {
    'train': {
        'background': background_symbols[:int(background_symbols.shape[0] * p_train)],
        'fch': fch_symbols[:int(fch_symbols.shape[0] * p_train)],
        'lvot': lvot_symbols[:int(lvot_symbols.shape[0] * p_train)],
        'three_vv': three_vv_symbols[:int(three_vv_symbols.shape[0] * p_train)],
        'three_vt': three_vt_symbols[:int(three_vt_symbols.shape[0] * p_train)],
    },
    'val': {
        'background': background_symbols[int(background_symbols.shape[0] * p_train):int(background_symbols.shape[0] * (p_train + p_val))],
        'fch': fch_symbols[int(fch_symbols.shape[0] * p_train):int(fch_symbols.shape[0] * (p_train + p_val))],
        'lvot': lvot_symbols[int(lvot_symbols.shape[0] * p_train):int(lvot_symbols.shape[0] * (p_train + p_val))],
        'three_vv': three_vv_symbols[int(three_vv_symbols.shape[0] * p_train):int(three_vv_symbols.shape[0] * (p_train + p_val))],
        'three_vt': three_vt_symbols[int(three_vt_symbols.shape[0] * p_train):int(three_vt_symbols.shape[0] * (p_train + p_val))],
    },
    'test': {
        'background': background_symbols[int(background_symbols.shape[0] * (p_train + p_val)):],
        'fch': fch_symbols[int(fch_symbols.shape[0] * (p_train + p_val)):],
        'lvot': lvot_symbols[int(lvot_symbols.shape[0] * (p_train + p_val)):],
        'three_vv': three_vv_symbols[int(three_vv_symbols.shape[0] * (p_train + p_val)):],
        'three_vt': three_vt_symbols[int(three_vt_symbols.shape[0] * (p_train + p_val)):],
    },
}


def get_data(phase: str):
    # pick 2 from background, 1 from fch, 1 from lvot, 1 from three_vv, 1 from three_vtd
    bg_indices = np.random.choice(len(data_dict[phase]['background']), size=2, replace=False)
    fch_index = np.random.choice(len(data_dict[phase]['fch']), size=1)
    lvot_index = np.random.choice(len(data_dict[phase]['lvot']), size=1)
    three_vv_index = np.random.choice(len(data_dict[phase]['three_vv']), size=1)
    three_vt_index = np.random.choice(len(data_dict[phase]['three_vt']), size=1)
    
    bg_start = data_dict[phase]['background'][bg_indices[0]]
    bg_end = data_dict[phase]['background'][bg_indices[1]]
    fch = data_dict[phase]['fch'][fch_index[0]]
    lvot = data_dict[phase]['lvot'][lvot_index[0]]
    three_vv = data_dict[phase]['three_vv'][three_vv_index[0]]
    three_vt = data_dict[phase]['three_vt'][three_vt_index[0]]
    
    print(bg_start.shape, bg_end.shape, fch.shape, lvot.shape, three_vv.shape, three_vt.shape)


if __name__ == "__main__":

    # Hyperparameters
    TANSITION_TIME = 0.5  # seconds
    FRAME_RATE = 60  # frames per second

    def get_portion():
        init_portion = {
            'start-background': 1.0,
            '4CH': 1.5,
            'LVOT': 1.5,
            '3VV': 1.5,
            '3VT': 1.5,
            'end-background': 1.0,
        }
        return_portion = {}
        for k, v in init_portion.items():
            # add gaussian noise to the value
            return_portion[k] = v + np.random.normal(0, 0.3)
        return return_portion

    time_portion = get_portion()




    print(time_portion)