from beatMnist_utils import get_element
import utils
import numpy as np
import random
import torch
from enum import Enum
from collections import OrderedDict
from morphInterp import morph_interp
from utils import save_video_opencv
from affine import AffineTransformer
from noise import add_noise2frames
import imageio, os
import json
from tqdm import tqdm
import argparse


# Hyper parameters
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
GEN_VIDEO_FPS = 60
TRANSITION_TIME = 0.5
SEED = 42

IN_H, IN_W = 28, 28
OUT_H, OUT_W = 112, 112
THETA_LEFT, THETA_RIGHT = -30, 30  # ratation
FEQ_ROT = 0.5  # Hz - frequency of rotation
MAX_SHIFT_PX = 40  # |dx| and |dy| will stay ≤ this
CUTOFF_HZ = 0.4  # controls smoothness (lower = smoother)
SCALE_MU = 1.5
SCALE_AMP = 0.5
SCALE_DELTA = 0.15
SCALE_F = 2.0

OUT_DIR = "gen_data/beatingMnistStandardNoise"
# OUT_DIR = "gen_data/beatingMnist2"


class CardiacViews:
    BACKGROUND = 0
    TRANSITION_1 = 5  # Background -> 4CH
    FCH = 1
    TRANSITION_2 = 6  # 4CH -> LVOT
    LVOT = 2
    TRANSITION_3 = 7  # LVOT -> 3VV
    THREE_VV = 3
    TRANSITION_4 = 8  # 3VV -> 3VT
    THREE_VT = 4
    TRANSITION_5 = 9  # 3VT -> Background


def get_time_portion():
    init_portion = OrderedDict({
        'start-background': 1.0,
        CardiacViews.TRANSITION_1: 0.3,
        CardiacViews.FCH: 1.5,
        CardiacViews.TRANSITION_2: 0.3,
        CardiacViews.LVOT: 1.5,
        CardiacViews.TRANSITION_3: 0.3,
        CardiacViews.THREE_VV: 1.5,
        CardiacViews.TRANSITION_4: 0.3,
        CardiacViews.THREE_VT: 1.5,
        CardiacViews.TRANSITION_5: 0.3,
        'end-background': 1.0,
    })
    return_portion = OrderedDict()
    for k, v in init_portion.items():
        if k in [
            CardiacViews.TRANSITION_1, CardiacViews.TRANSITION_2, 
            CardiacViews.TRANSITION_3, CardiacViews.TRANSITION_4, 
            CardiacViews.TRANSITION_5]:
            return_portion[k] = v + np.random.normal(0, 0.03)
        else:
            return_portion[k] = v + np.random.normal(0, 0.03)
    return return_portion


def assemble_video(
        time_portion: OrderedDict,
        dataset: dict,
        fps: int,
        ) -> tuple:
    frames_count = OrderedDict()
    for k, v in time_portion.items():
        frames_count[k] = int(v * fps)

    # init the final frames dict
    frames_collection = OrderedDict()
    for k, v in frames_count.items():
        frames_collection[k] = []

    for k, v in frames_count.items():
        if k in ['start-background', 'end-background']:
            sample_num = len(dataset[CardiacViews.BACKGROUND])
            mid = dataset[CardiacViews.BACKGROUND][
                np.random.choice(sample_num)
                ]
            frames_collection[k].extend([dataset[CardiacViews.BACKGROUND][
                np.random.choice(sample_num)
                ]] * frames_count[k])
        elif k in [
            CardiacViews.TRANSITION_1, CardiacViews.TRANSITION_2, 
            CardiacViews.TRANSITION_3, CardiacViews.TRANSITION_4, 
            CardiacViews.TRANSITION_5]:
            continue
        else:
            sample_num = len(dataset[k])
            
            frames_collection[k].extend([dataset[k][
                np.random.choice(sample_num)
            ]] * frames_count[k]
            )
    
    # 2nd loop to handling the transition frames
    for k, v in frames_count.items():
        condition_1 = (
            k == CardiacViews.TRANSITION_1 and \
            'start-background' in frames_count.keys() and \
            CardiacViews.FCH in frames_count.keys()
        )
        condition_2 = (
            k == CardiacViews.TRANSITION_2 and \
            CardiacViews.FCH in frames_count.keys() and \
            CardiacViews.LVOT in frames_count.keys()
        )
        condition_3 = (
            k == CardiacViews.TRANSITION_3 and \
            CardiacViews.LVOT in frames_count.keys() and \
            CardiacViews.THREE_VV in frames_count.keys()
        )
        condition_4 = (
            k == CardiacViews.TRANSITION_4 and \
            CardiacViews.THREE_VV in frames_count.keys() and \
            CardiacViews.THREE_VT in frames_count.keys()
        )
        condition_5 = (
            k == CardiacViews.TRANSITION_5 and \
            CardiacViews.THREE_VT in frames_count.keys() and \
            'end-background' in frames_count.keys()
        )
        if condition_1:
            frames_collection[k].extend(
                morph_interp(
                    frames_collection['start-background'][0],
                    frames_collection[CardiacViews.FCH][0],
                    frames_count[k]
                )
            )
        elif condition_2:
            frames_collection[k].extend(
                morph_interp(
                    frames_collection[CardiacViews.FCH][0],
                    frames_collection[CardiacViews.LVOT][0],
                    frames_count[k]
                )
            )
        elif condition_3:
            frames_collection[k].extend(
                morph_interp(
                    frames_collection[CardiacViews.LVOT][0],
                    frames_collection[CardiacViews.THREE_VV][0],
                    frames_count[k]
                )
            )
        elif condition_4:
            frames_collection[k].extend(
                morph_interp(
                    frames_collection[CardiacViews.THREE_VV][0],
                    frames_collection[CardiacViews.THREE_VT][0],
                    frames_count[k]
                )
            )
        elif condition_5:
            frames_collection[k].extend(
                morph_interp(
                    frames_collection[CardiacViews.THREE_VT][0],
                    frames_collection['end-background'][0],
                    frames_count[k]
                )
            )
        else:
            continue
        
        video_annotations = []
        final_frames = []
        for k, v in frames_collection.items():
            final_frames.extend(v)
            if k in ['start-background', 'end-background']:
                video_annotations.extend([CardiacViews.BACKGROUND] * len(v))
            else:
                video_annotations.extend([k] * len(v))

    return final_frames, video_annotations
    

def gen_single_video(
        dataset: dict,
        fps: int,
        save_path: str | None,
        video_index: int = 0,
        time_portion: OrderedDict = None,
        ):
    '''
    Generate a single video with a given time portion.
    Return the video frame arrays and the labeling of the portion as well.
    '''
    # Generate time portion if not provided (for multiprocessing)
    if time_portion is None:
        time_portion = get_time_portion()
    
    video, annotations = assemble_video(
        dataset = dataset,
        fps = fps,
        time_portion = time_portion,
    )

    # Create unique seed for each video to ensure different trajectories
    video_seed = SEED + video_index * 1000  # Large multiplier to avoid seed collision
    
    aft = AffineTransformer(
        rotation_theta_left = THETA_LEFT,
        rotation_theta_right = THETA_RIGHT,
        rotation_f_rot = FEQ_ROT,  # rotation frequency in Hz
        offset_max_px = MAX_SHIFT_PX,   # maximum offset in pixels
        offset_cutoff_hz = CUTOFF_HZ,
        scale_mu = SCALE_MU,
        scale_amp = SCALE_AMP,
        scale_delta = SCALE_DELTA,  # ±5 % jitter
        scale_f = SCALE_F,  # Hz
        seed = video_seed,
    )
    video = aft.transform(
        list_of_images = video,
        out_shape = (OUT_H, OUT_W),
        order = 1,
        fps = fps
    )
    video = add_noise2frames(
        video,
        gaussian_sigma   = 1.40,     # σ of additive N(0,σ²) noise
        sp_prob          = 0.15,     # salt-&-pepper pixel probability
        brightness_delta = 0.20,     # ±fractional shift
        contrast_low     = 0.80,     # contrast factor drawn ∈ [low, high]
        contrast_high    = 1.20,
        seed             = video_seed,  # Use unique seed for each video
    )
    
    video = [utils.floatImage2uint8(frame) for frame in video]

    if save_path:
        imageio.mimsave(save_path, video, fps=fps)
        # save annotation list as json file
        with open(save_path.replace('.mp4', '.json'), 'w') as f:
            json.dump(annotations, f)
        
def generate_single_video_wrapper(args):
    """Wrapper function for multiprocessing compatibility."""
    dataset, fps, save_path, video_index = args
    return gen_single_video(
        dataset=dataset,
        fps=fps,
        save_path=save_path,
        video_index=video_index,
        time_portion=None  # Will be generated inside the function
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default="./" help='Output directory')
    args = parser.parse_args()
    
    # Set global seeds for reproducibility
    np.random.seed(SEED)
    random.seed(SEED)

    # Split dataset
    background_symbols = np.concatenate([get_element('Z')[0], get_element('z')[0]], axis=0)
    fch_symbols = get_element('Q')[0]
    lvot_symbols = get_element('0')[0]
    three_vv_symbols = get_element('I')[0]
    three_vt_symbols = get_element('L')[0]

    data_dict = {
    'train': {
        CardiacViews.BACKGROUND: background_symbols[:int(background_symbols.shape[0] * TRAIN_RATIO)],
        CardiacViews.FCH: fch_symbols[:int(fch_symbols.shape[0] * TRAIN_RATIO)],
        CardiacViews.LVOT: lvot_symbols[:int(lvot_symbols.shape[0] * TRAIN_RATIO)],
        CardiacViews.THREE_VV: three_vv_symbols[:int(three_vv_symbols.shape[0] * TRAIN_RATIO)],
        CardiacViews.THREE_VT: three_vt_symbols[:int(three_vt_symbols.shape[0] * TRAIN_RATIO)],
    },
    'val': {
        CardiacViews.BACKGROUND: background_symbols[int(background_symbols.shape[0] * TRAIN_RATIO):int(background_symbols.shape[0] * (TRAIN_RATIO + VAL_RATIO))],
        CardiacViews.FCH: fch_symbols[int(fch_symbols.shape[0] * TRAIN_RATIO):int(fch_symbols.shape[0] * (TRAIN_RATIO + VAL_RATIO))],
        CardiacViews.LVOT: lvot_symbols[int(lvot_symbols.shape[0] * TRAIN_RATIO):int(lvot_symbols.shape[0] * (TRAIN_RATIO + VAL_RATIO))],
        CardiacViews.THREE_VV: three_vv_symbols[int(three_vv_symbols.shape[0] * TRAIN_RATIO):int(three_vv_symbols.shape[0] * (TRAIN_RATIO + VAL_RATIO))],
        CardiacViews.THREE_VT: three_vt_symbols[int(three_vt_symbols.shape[0] * TRAIN_RATIO):int(three_vt_symbols.shape[0] * (TRAIN_RATIO + VAL_RATIO))],
    },
    'test': {
        CardiacViews.BACKGROUND: background_symbols[int(background_symbols.shape[0] * (TRAIN_RATIO + VAL_RATIO)):],
        CardiacViews.FCH: fch_symbols[int(fch_symbols.shape[0] * (TRAIN_RATIO + VAL_RATIO)):],
        CardiacViews.LVOT: lvot_symbols[int(lvot_symbols.shape[0] * (TRAIN_RATIO + VAL_RATIO)):],
        CardiacViews.THREE_VV: three_vv_symbols[int(three_vv_symbols.shape[0] * (TRAIN_RATIO + VAL_RATIO)):],
        CardiacViews.THREE_VT: three_vt_symbols[int(three_vt_symbols.shape[0] * (TRAIN_RATIO + VAL_RATIO)):],
    },
    }
    
    # use multiprocessing to generate videos
    import multiprocessing as mp
    
    
    # Prepare arguments for multiprocessing

    tasks = [
        # ('train', 1, 100),  # phase, start_index, video_num
        ('train', 1, 10000),  # phase, start_index, video_num
        ('val', 100001, 500),
        ('test',200001, 2000),
    ]

    for phase, start_index, num_videos in tasks:
        args_list = [
            (data_dict[phase], GEN_VIDEO_FPS, f"{args.out_dir}/{phase}/video_{(i+start_index):07d}.mp4", i) 
            for i in range(num_videos)
        ]
        os.makedirs(f"{args.out_dir}/{phase}", exist_ok=True)
        print(f"Generating {num_videos} videos using {mp.cpu_count()} processes...")
        
        with mp.Pool(processes=min(10, mp.cpu_count())) as pool:
            # Use map instead of starmap with wrapper function
            results = []
            for result in tqdm(pool.imap(generate_single_video_wrapper, args_list), total=num_videos):
                results.append(result)
        
        print(f"Successfully generated {len(results)} videos")

    # for i in tqdm(range(50)):
    # # for i in range(50):
    #     os.makedirs(f"{OUT_DIR}", exist_ok=True)

    #     gen_single_video(
    #         time_portion = get_time_portion(),
    #         fps = GEN_VIDEO_FPS,
    #         dataset = data_dict['train'],
    #         save_path = f"{OUT_DIR}/single_video_{i}.mp4",
    #         video_index = i,
    #         # marker = False,
    #     )