#!/usr/bin/env python3
"""
Test script for trajectory extraction from VIPE SLAM output
"""

import os
import sys
sys.path.append('/home/hongyuan/world-decoder/vipe')

from extrinsic_path.extract_extrinsic_path import process_vipe_segments_for_trajectories

def test_single_video(video_path):
    """Test trajectory extraction on a single video."""
    print("="*60)
    print("Testing Single Video Trajectory Extraction with VIPE SLAM")
    print("="*60)
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Process the video
    video_dir = os.path.dirname(video_path)
    
    # Temporarily move video to temp dir to process just this one
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_video = os.path.join(tmpdir, os.path.basename(video_path))
        shutil.copy(video_path, temp_video)
        
        # Also copy metadata if it exists
        metadata_path = video_path.replace('.mp4', '_metadata.json').replace('.MP4', '_metadata.json')
        if os.path.exists(metadata_path):
            temp_metadata = os.path.join(tmpdir, os.path.basename(metadata_path))
            shutil.copy(metadata_path, temp_metadata)
        
        from extrinsic_path.extract_extrinsic_path import process_vipe_segments_for_trajectories
        
        trajectories, orientations, trajectories_2d, info = process_vipe_segments_for_trajectories(
            segments_dir=tmpdir
        )
        
        if trajectories:
            print("\n" + "="*50)
            print("RESULTS:")
            print("="*50)
            for segment_info in info:
                print(f"Segment: {segment_info['segment_name']}")
                print(f"Number of frames: {segment_info['num_frames']}")
                print(f"Trajectory length: {segment_info['trajectory_length_meters']:.2f} meters")
                print(f"Scale factor: {segment_info['scale_factor']:.4f}")
            print("="*50)
        else:
            print("Failed to extract trajectory")

def test_all_segments(segments_dir="/home/hongyuan/world-decoder/dataset/video_samples"):
    """Test trajectory extraction on all segments in a directory."""
    print("="*60)
    print("Testing All Segments Trajectory Extraction with VIPE SLAM")
    print("="*60)
    
    if not os.path.exists(segments_dir):
        print(f"Error: Segments directory not found: {segments_dir}")
        return
    
    from extrinsic_path.extract_extrinsic_path import process_vipe_segments_for_trajectories
    
    trajectories, orientations, trajectories_2d, info = process_vipe_segments_for_trajectories(
        segments_dir=segments_dir
    )
    
    if trajectories:
        print("\n" + "="*50)
        print(f"Successfully processed {len(trajectories)} segments")
        print("="*50)
    else:
        print("No segments were processed successfully")

def main_menu():
    """Simple menu for testing."""
    print("VIPE SLAM Trajectory Extraction Test Menu")
    print("="*40)
    print("1. Test single video")
    print("2. Test all segments in directory")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            video_path = input("Enter video path: ").strip()
            try:
                test_single_video(video_path)
            except Exception as e:
                print(f"Error in single video test: {e}")
                import traceback
                traceback.print_exc()
        elif choice == '2':
            segments_dir = input("Enter segments directory [/home/hongyuan/world-decoder/dataset/video_samples]: ").strip()
            if not segments_dir:
                segments_dir = "/home/hongyuan/world-decoder/dataset/video_samples"
            try:
                test_all_segments(segments_dir)
            except Exception as e:
                print(f"Error in all segments test: {e}")
                import traceback
                traceback.print_exc()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    print("VIPE SLAM Trajectory Extraction Test")
    print("This script tests the trajectory extraction functionality")
    print("using VIPE's SLAM pipeline")
    print()
    
    main_menu()

