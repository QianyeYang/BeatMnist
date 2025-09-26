import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import cv2

source_1_dir = "gen_data/beatingMnist"
source_2_dir = "gen_data/beatingMnist2"

def load_videos(dir1, dir2, num_videos=5):
    """Load the first num_videos from both directories."""
    
    # Get sorted file lists to ensure consistent ordering
    files_1 = sorted([f for f in os.listdir(dir1) if f.endswith('.mp4')])[:num_videos]
    files_2 = sorted([f for f in os.listdir(dir2) if f.endswith('.mp4')])[:num_videos]
    
    videos_1 = []
    videos_2 = []
    
    print(f"Loading {num_videos} videos from each directory...")
    
    for i in range(min(len(files_1), len(files_2), num_videos)):
        print(f"Loading video {i+1}/{num_videos}: {files_1[i]} and {files_2[i]}")
        
        # Load videos
        video_1 = imageio.mimread(os.path.join(dir1, files_1[i]))
        video_2 = imageio.mimread(os.path.join(dir2, files_2[i]))
        
        # Convert to numpy arrays and ensure consistent format
        video_1 = np.array(video_1)
        video_2 = np.array(video_2)
        
        # Handle grayscale conversion if needed
        if len(video_1.shape) == 4 and video_1.shape[-1] == 3:
            video_1 = np.mean(video_1, axis=-1).astype(np.uint8)
        if len(video_2.shape) == 4 and video_2.shape[-1] == 3:
            video_2 = np.mean(video_2, axis=-1).astype(np.uint8)
            
        videos_1.append(video_1)
        videos_2.append(video_2)
    
    return videos_1, videos_2, files_1[:num_videos], files_2[:num_videos]

def create_side_by_side_comparison(video_1, video_2, title_1="Video 1", title_2="Video 2"):
    """Create a side-by-side comparison of two videos."""
    
    # Ensure both videos have the same number of frames
    min_frames = min(len(video_1), len(video_2))
    video_1 = video_1[:min_frames]
    video_2 = video_2[:min_frames]
    
    # Get dimensions
    h1, w1 = video_1[0].shape[:2]
    h2, w2 = video_2[0].shape[:2]
    
    # Create combined frames
    combined_frames = []
    
    # Calculate padding to make heights equal
    max_height = max(h1, h2)
    pad_1 = (max_height - h1) // 2
    pad_2 = (max_height - h2) // 2
    
    for frame_1, frame_2 in zip(video_1, video_2):
        # Pad frames to same height
        if pad_1 > 0:
            frame_1 = np.pad(frame_1, ((pad_1, max_height - h1 - pad_1), (0, 0)), mode='constant', constant_values=0)
        if pad_2 > 0:
            frame_2 = np.pad(frame_2, ((pad_2, max_height - h2 - pad_2), (0, 0)), mode='constant', constant_values=0)
        
        # Add separator line
        separator = np.ones((max_height, 3), dtype=np.uint8) * 255
        
        # Combine horizontally
        combined_frame = np.hstack([frame_1, separator, frame_2])
        combined_frames.append(combined_frame)
    
    return np.array(combined_frames)

def create_two_row_grid(videos_1, videos_2, filenames_1, filenames_2, output_path="video_comparison.mp4"):
    """Create a 2-row grid: top row = videos_1, bottom row = videos_2."""
    
    if not videos_1 or not videos_2:
        print("No videos found in one or both directories!")
        return
    
    num_videos = min(len(videos_1), len(videos_2))
    print(f"Creating 2-row grid with {num_videos} videos per row...")
    
    # Find minimum number of frames across all videos
    all_videos = videos_1 + videos_2
    min_frames = min(len(video) for video in all_videos)
    print(f"Using {min_frames} frames (minimum across all videos)")
    
    # Trim all videos to the same length
    videos_1 = [video[:min_frames] for video in videos_1]
    videos_2 = [video[:min_frames] for video in videos_2]
    
    # Get frame dimensions and find max dimensions
    all_videos_trimmed = videos_1 + videos_2
    heights = [video[0].shape[0] for video in all_videos_trimmed]
    widths = [video[0].shape[1] for video in all_videos_trimmed]
    max_height = max(heights)
    max_width = max(widths)
    
    print(f"Standardizing all frames to {max_height}x{max_width}")
    
    # Function to pad and resize frame to standard size
    def standardize_frame(frame, target_h, target_w):
        h, w = frame.shape[:2]
        
        # Pad to make it square-ish first
        if h < target_h:
            pad_h = (target_h - h) // 2
            frame = np.pad(frame, ((pad_h, target_h - h - pad_h), (0, 0)), mode='constant', constant_values=0)
        elif h > target_h:
            # Crop if too tall
            start = (h - target_h) // 2
            frame = frame[start:start + target_h]
            
        if w < target_w:
            pad_w = (target_w - w) // 2
            frame = np.pad(frame, ((0, 0), (pad_w, target_w - w - pad_w)), mode='constant', constant_values=0)
        elif w > target_w:
            # Crop if too wide
            start = (w - target_w) // 2
            frame = frame[:, start:start + target_w]
            
        return frame
    
    # Standardize all videos
    standardized_videos_1 = []
    standardized_videos_2 = []
    
    for video in videos_1:
        standardized_video = [standardize_frame(frame, max_height, max_width) for frame in video]
        standardized_videos_1.append(standardized_video)
        
    for video in videos_2:
        standardized_video = [standardize_frame(frame, max_height, max_width) for frame in video]
        standardized_videos_2.append(standardized_video)
    
    # Create grid layout: 2 rows x num_videos columns
    rows, cols = 2, num_videos
    separator_width = 5  # pixels between videos
    
    # Calculate final grid dimensions
    grid_height = rows * max_height + (rows - 1) * separator_width
    grid_width = cols * max_width + (cols - 1) * separator_width
    
    print(f"Final grid dimensions: {grid_height}x{grid_width}")
    
    # Create grid frames
    grid_frames = []
    
    for frame_idx in range(min_frames):
        if frame_idx % 10 == 0:  # Progress indicator
            print(f"Processing frame {frame_idx + 1}/{min_frames}")
            
        # Create empty grid frame
        grid_frame = np.zeros((grid_height, grid_width), dtype=np.uint8)
        
        # Fill top row (videos_1)
        for col_idx in range(num_videos):
            if col_idx < len(standardized_videos_1):
                frame = standardized_videos_1[col_idx][frame_idx]
                
                # Calculate position in grid
                y_start = 0
                y_end = max_height
                x_start = col_idx * (max_width + separator_width)
                x_end = x_start + max_width
                
                grid_frame[y_start:y_end, x_start:x_end] = frame
        
        # Fill bottom row (videos_2)
        for col_idx in range(num_videos):
            if col_idx < len(standardized_videos_2):
                frame = standardized_videos_2[col_idx][frame_idx]
                
                # Calculate position in grid
                y_start = max_height + separator_width
                y_end = y_start + max_height
                x_start = col_idx * (max_width + separator_width)
                x_end = x_start + max_width
                
                grid_frame[y_start:y_end, x_start:x_end] = frame
        
        grid_frames.append(grid_frame)
    
    # Save the grid comparison video
    print(f"Saving comparison video to {output_path}...")
    imageio.mimsave(output_path, grid_frames, fps=30)
    print(f"Comparison video saved successfully!")
    print(f"Grid layout: 2 rows x {num_videos} columns")
    print(f"Top row: {source_1_dir}")
    print(f"Bottom row: {source_2_dir}")
    
    return grid_frames

if __name__ == "__main__":
    # Check if directories exist
    if not os.path.exists(source_1_dir):
        print(f"Error: Directory {source_1_dir} does not exist!")
        exit(1)
    if not os.path.exists(source_2_dir):
        print(f"Error: Directory {source_2_dir} does not exist!")
        exit(1)
    
    # Load videos from both directories
    videos_1, videos_2, files_1, files_2 = load_videos(source_1_dir, source_2_dir, num_videos=5)
    
    if not videos_1 or not videos_2:
        print("No videos found to compare!")
        exit(1)
    
    # Create comparison
    print(f"\nComparing videos:")
    print(f"Directory 1 ({source_1_dir}): {len(videos_1)} videos")
    print(f"Directory 2 ({source_2_dir}): {len(videos_2)} videos")
    
    # Create and save the comparison video
    comparison_frames = create_two_row_grid(videos_1, videos_2, files_1, files_2, 
                                               output_path="video_comparison_grid.mp4")
    
    # Also create individual side-by-side comparisons
    print("\nCreating individual comparisons...")
    for i, (v1, v2, f1, f2) in enumerate(zip(videos_1, videos_2, files_1, files_2)):
        comparison = create_side_by_side_comparison(v1, v2, f1, f2)
        output_name = f"comparison_{i:02d}_{os.path.splitext(f1)[0]}_vs_{os.path.splitext(f2)[0]}.mp4"
        imageio.mimsave(output_name, comparison, fps=30)
        print(f"Saved: {output_name}")
    
    print("\nVideo comparison complete!")
    print(f"Main comparison grid: video_comparison_grid.mp4")
    print(f"Individual comparisons: comparison_*.mp4")
