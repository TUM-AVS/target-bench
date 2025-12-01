#!/usr/bin/env python3
"""
Simple example demonstrating VIPE trajectory extraction with frame sampling.

This script shows how to:
1. Load a video file and extract evenly-spaced frames
2. Create a sampled video from the extracted frames
3. Run VIPE SLAM to estimate camera poses on the sampled video
4. Extract and save the trajectory
5. Visualize the results

The script uses intelligent adaptive sampling based on video length:
- Short videos (≤200 frames): Use ALL frames for best accuracy
- Medium videos (200-500): Use 50% of frames  
- Long videos (>500): Use 30% of frames (min 50, max 200)

This balances efficient processing with sufficient frame overlap for robust SLAM.
"""

import os
import sys
import cv2
import numpy as np
import shutil
from pathlib import Path

# Add VIPE to path
sys.path.append('/home/hongyuan/world-decoder/vipe')
sys.path.append('/home/hongyuan/world-decoder/world-decoder_code_vipe')

def extract_evenly_spaced_frames(video_path, num_frames=25):
    """
    Extract evenly spaced frames from a video.
    
    Args:
        video_path: Path to input video file
        num_frames: Number of frames to extract (default: 25)
        
    Returns:
        frames: List of numpy arrays containing the extracted frames
        frame_indices: List of frame indices that were extracted
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError(f"Video has 0 frames: {video_path}")
    
    # Calculate evenly spaced frame indices
    if num_frames >= total_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print(f"Warning: Could not read frame {idx}")
    
    cap.release()
    print(f"Extracted {len(frames)} frames from {total_frames} total frames")
    print(f"Frame indices: {frame_indices}")
    
    return frames, frame_indices

def create_video_from_frames(frames, output_path, fps=30):
    """
    Create a video from a list of frames.
    
    Args:
        frames: List of numpy arrays (frames)
        output_path: Path to save the output video
        fps: Frames per second for the output video
    """
    if not frames:
        raise ValueError("No frames to create video")
    
    height, width = frames[0].shape[:2]
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"Created video: {output_path}")
    print(f"  - Size: {width}x{height}")
    print(f"  - Frames: {len(frames)}")
    print(f"  - FPS: {fps}")
    
    return output_path

def example_single_video():
    """
    Example: Process a single video file and extract trajectory.
    Uses intelligent adaptive sampling based on video length and creates
    a sampled video optimized for SLAM processing.
    """
    print("=" * 60)
    print("VIPE Trajectory Extraction - Single Video Example")
    print("=" * 60)
    
    # Import the trajectory extraction functions
    from extrinsic_path.extract_extrinsic_path import (
        process_vipe_segments_for_trajectories,
        save_trajectory_data,
        visualize_trajectory_with_gt
    )
    
    # Configuration
    video_path = "/home/hongyuan/world-decoder/dataset/Benchmark/sample_087_custom_segment_057_358"  # Change this!
    output_dir = "/home/hongyuan/world-decoder/evaluation_results/vipe"
    
    # Frame sampling options:
    # - None: Adaptive sampling based on video length - RECOMMENDED
    #   * Short (≤200): Use all frames
    #   * Medium (200-500): Use 50% of frames
    #   * Long (>500): Use 30% of frames (min 50, max 200)
    # - -1: Force use of ALL frames from the original video
    # - Integer: Use exactly that many frames (e.g., 100)
    num_frames_to_sample = 9  # Extract only 9 frames evenly spaced

    # Check if video directory exists
    if not os.path.exists(video_path):
        print(f"Please update video_path in the script!")
        print(f"Current path (does not exist): {video_path}")
        return
    
    # Use video_path directly (it's already a directory containing the video)
    video_dir = video_path if os.path.isdir(video_path) else os.path.dirname(video_path)
    
    print(f"\nInput video directory: {video_dir}")
    print(f"Output directory: {output_dir}")
    
    # Step 0: Find the original video file and create sampled version
    print("\n[0/4] Creating sampled video with evenly-spaced frames...")
    
    # Find video files in the directory
    video_extensions = ['.mp4', '.MP4', '.avi', '.mov']
    original_video = None
    for ext in video_extensions:
        video_files = list(Path(video_dir).glob(f"*{ext}"))
        if video_files:
            original_video = str(video_files[0])
            break
    
    if not original_video:
        print(f"❌ No video file found in {video_dir}")
        return
    
    print(f"Found original video: {original_video}")
    
    # Determine optimal number of frames to sample
    cap = cv2.VideoCapture(original_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if num_frames_to_sample is None:
        # Adaptive sampling strategy:
        # - For short videos (<= 200 frames): use ALL frames (best accuracy)
        # - For medium videos (200-500): use 50% of frames
        # - For long videos (>500): use 30% with bounds [50, 200]
        if total_frames <= 200:
            num_frames_to_sample = total_frames
            print(f"Using ALL {total_frames} frames (video is short, no sampling needed)")
        elif total_frames <= 500:
            num_frames_to_sample = max(100, int(total_frames * 0.5))
            print(f"Auto-selected {num_frames_to_sample} frames from {total_frames} total (50% sampling)")
            print(f"Frame spacing: approximately every {max(1, total_frames // num_frames_to_sample)} frames")
        else:
            num_frames_to_sample = max(50, min(200, int(total_frames * 0.3)))
            print(f"Auto-selected {num_frames_to_sample} frames from {total_frames} total (30% sampling)")
            print(f"Frame spacing: approximately every {max(1, total_frames // num_frames_to_sample)} frames")
    elif num_frames_to_sample == -1:
        # Use all frames (no sampling)
        num_frames_to_sample = total_frames
        print(f"Using ALL {total_frames} frames (no sampling)")
    
    # Extract evenly spaced frames
    try:
        frames, frame_indices = extract_evenly_spaced_frames(original_video, num_frames=num_frames_to_sample)
    except Exception as e:
        print(f"❌ Failed to extract frames: {e}")
        return
    
    # Create sampled video in a temporary directory
    sampled_video_dir = os.path.join(output_dir, "sampled_videos")
    os.makedirs(sampled_video_dir, exist_ok=True)
    
    video_basename = os.path.basename(video_dir)
    sampled_video_path = os.path.join(sampled_video_dir, f"{video_basename}_sampled.mp4")
    
    try:
        create_video_from_frames(frames, sampled_video_path, fps=30)
    except Exception as e:
        print(f"❌ Failed to create sampled video: {e}")
        return
    
    # Create a temporary directory with the sampled video for VIPE processing
    temp_segment_dir = os.path.join(sampled_video_dir, video_basename)
    os.makedirs(temp_segment_dir, exist_ok=True)
    
    # Copy or move the sampled video to the segment directory with expected naming
    final_video_path = os.path.join(temp_segment_dir, f"{video_basename}_video.mp4")
    shutil.copy(sampled_video_path, final_video_path)
    print(f"Prepared sampled video for VIPE: {final_video_path}")
    
    # Step 1: Process the sampled video with VIPE SLAM
    print("\n[1/4] Running VIPE SLAM on sampled video...")
    trajectories, orientations, trajectories_2d, segment_info = process_vipe_segments_for_trajectories(
        segments_dir=temp_segment_dir
    )
    
    if not trajectories:
        print("❌ Failed to extract trajectory")
        return
    
    trajectory = trajectories[0]
    orientation = orientations[0]
    info = segment_info[0]
    
    # Check if trajectory contains NaN values (SLAM failure)
    if np.isnan(trajectory).any():
        print("\n⚠️  Warning: SLAM produced invalid trajectory (contains NaN values)")
        print("This can happen when:")
        print("  - Frames are too far apart (insufficient overlap)")
        print("  - Scene lacks sufficient texture/features")
        print("  - Camera motion is too large between frames")
        print("  - Lighting conditions change significantly between frames")
        print("\nRecommendation: Try one of the following:")
        print("  - Use more frames (increase sampling percentage)")
        print("  - Process the full video without sampling")
        print("  - Check if the video has sufficient visual features")
        print("Continuing to save results anyway...")
    
    # Step 2: Save the trajectory
    print("\n[2/4] Saving trajectory data...")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, info['segment_name'])
    save_trajectory_data(trajectory, orientation, output_path, info)
    
    # Step 3: Visualize
    print("\n[3/4] Creating visualization...")
    visualize_trajectory_with_gt(
        trajectory,
        gt_trajectory_2d=None,  # No ground truth in this example
        output_path=output_path,
        title=f"VIPE SLAM: {info['segment_name']} (Sampled {num_frames_to_sample} frames)"
    )
    
    # Step 4: Cleanup temporary files (optional)
    print("\n[4/4] Cleaning up temporary files...")
    try:
        # Keep the sampled video but remove the temporary segment directory
        if os.path.exists(temp_segment_dir):
            shutil.rmtree(temp_segment_dir)
            print(f"Cleaned up temporary directory: {temp_segment_dir}")
        print(f"Sampled video saved at: {sampled_video_path}")
    except Exception as e:
        print(f"Warning: Could not clean up temporary files: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("✓ Processing Complete!")
    print("=" * 60)
    print(f"Original video: {original_video}")
    print(f"Sampled video: {sampled_video_path}")
    print(f"Frames sampled: {len(frames)} frames evenly spaced")
    print(f"Sample indices: {frame_indices}")
    print(f"\nSegment: {info['segment_name']}")
    print(f"Number of poses: {info['num_frames']}")
    print(f"Trajectory length: {info['trajectory_length_meters']:.2f} meters")
    print(f"\nOutput files:")
    print(f"  - {output_path}_trajectory.npy")
    print(f"  - {output_path}_trajectory.csv")
    print(f"  - {output_path}_trajectory.json")
    print(f"  - {output_path}_trajectory_plot.png")
    print("=" * 60)

def example_batch_processing():
    """
    Example: Process multiple videos in a directory.
    """
    print("=" * 60)
    print("VIPE Trajectory Extraction - Batch Processing Example")
    print("=" * 60)
    
    from extrinsic_path.extract_extrinsic_path import (
        process_vipe_segments_for_trajectories,
        save_trajectory_data,
        create_trajectory_summary_report
    )
    
    # Configuration
    videos_dir = "/home/hongyuan/world-decoder/dataset/Benchmark/sample_087_custom_segment_057_358"  # Change this!
    output_dir = "/home/hongyuan/world-decoder/evaluation_results/vipe"
    
    # Check if directory exists
    if not os.path.exists(videos_dir):
        print(f"Please update videos_dir in the script!")
        print(f"Current path (does not exist): {videos_dir}")
        return
    
    print(f"\nInput directory: {videos_dir}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Process all videos
    print("\n[1/2] Running VIPE SLAM on all videos...")
    trajectories, orientations, trajectories_2d, segment_info = process_vipe_segments_for_trajectories(
        segments_dir=videos_dir
    )
    
    if not trajectories:
        print("❌ No trajectories extracted")
        return
    
    print(f"✓ Successfully processed {len(trajectories)} videos")
    
    # Step 2: Save all trajectories
    print("\n[2/2] Saving trajectories...")
    os.makedirs(output_dir, exist_ok=True)
    
    for trajectory, orientation, info in zip(trajectories, orientations, segment_info):
        output_path = os.path.join(output_dir, info['segment_name'])
        save_trajectory_data(trajectory, orientation, output_path, info)
        print(f"  ✓ Saved {info['segment_name']}")
    
    # Create summary report
    report_path = os.path.join(output_dir, "summary")
    create_trajectory_summary_report(segment_info, report_path)
    
    print("\n" + "=" * 60)
    print("✓ Batch Processing Complete!")
    print("=" * 60)
    print(f"Processed {len(trajectories)} videos")
    print(f"Output directory: {output_dir}")
    print(f"Summary report: {report_path}_trajectory_report.txt")
    print("=" * 60)

def main():
    """Main menu for examples."""
    print("\n" + "=" * 60)
    print("VIPE Trajectory Extraction - Examples")
    print("=" * 60)
    print("\n1. Single video example")
    print("2. Batch processing example")
    print("3. Exit")
    
    choice = input("\nSelect example (1-3): ").strip()
    
    if choice == '1':
        example_single_video()
    elif choice == '2':
        example_batch_processing()
    elif choice == '3':
        print("Exiting...")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    print("""
VIPE Trajectory Extraction - Usage Examples

This script demonstrates how to use the VIPE trajectory extraction system
with automatic frame sampling.

Before running:
1. Update the video paths in the example functions
2. Ensure VIPE is installed and accessible
3. Have CUDA available for SLAM processing

The system will:
- Intelligently sample frames based on video length:
  * Short videos (≤200): Use all frames
  * Medium videos (200-500): Use 50% of frames
  * Long videos (>500): Use 30% of frames
- Create a sampled video optimized for SLAM
- Run visual SLAM with robust configuration
- Extract camera trajectories
- Save results in multiple formats (NPY, CSV, JSON)
- Generate visualizations
    """)
    
    main()

