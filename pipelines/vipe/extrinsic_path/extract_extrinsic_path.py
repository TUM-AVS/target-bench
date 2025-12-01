"""
Extract extrinsic path and build trajectories from VIPE SLAM output.

This module processes video segments using VIPE's SLAM pipeline and extracts
camera trajectories with real metric scale.

Key functionality:
1. Processes video segments using VIPE SLAM pipeline
2. Extracts camera trajectories from SLAM output poses
3. Projects trajectories onto the X-Y plane for visualization
4. Creates comprehensive visualizations and reports
5. Handles multiple segments and provides statistical analysis

Main functions:
- process_vipe_segments_for_trajectories(): Process all segments and extract trajectories
- extract_trajectory_from_poses(): Convert pose matrices to trajectory points
- visualize_multiple_trajectories(): Create visualizations for multiple segments
- main(): Process all segments and generate outputs

Usage:
- Run directly: python extract_extrinsic_path.py
- Or use test script: python test_trajectory_extraction.py
"""

import torch
import numpy as np
import os
import pickle
import glob
import pandas as pd
import yaml
import json
from pathlib import Path
import sys

# Add the vipe module to path
sys.path.append('/home/hongyuan/world-decoder/vipe')

from vipe.slam.system import SLAMSystem
from vipe.streams.base import StreamList
from vipe.pipeline import make_pipeline
from omegaconf import DictConfig, OmegaConf
import hydra

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    # q = [qw, qx, qy, qz]
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    R = np.zeros(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2*(qy**2 + qz**2)
    R[..., 0, 1] = 2*(qx*qy - qw*qz)
    R[..., 0, 2] = 2*(qx*qz + qw*qy)
    R[..., 1, 0] = 2*(qx*qy + qw*qz)
    R[..., 1, 1] = 1 - 2*(qx**2 + qz**2)
    R[..., 1, 2] = 2*(qy*qz - qw*qx)
    R[..., 2, 0] = 2*(qx*qz - qw*qy)
    R[..., 2, 1] = 2*(qy*qz + qw*qx)
    R[..., 2, 2] = 1 - 2*(qx**2 + qy**2)
    return R

def extract_trajectory_from_poses(poses):
    """
    Extract camera trajectory from VIPE SLAM poses.
    
    Args:
        poses: torch.Tensor of shape (num_frames, 7) or (1, num_frames, 7)
               Format: [qw, qx, qy, qz, tx, ty, tz] per frame
        
    Returns:
        trajectory: numpy array of shape (num_frames, 3) - camera positions in meters
        orientations: numpy array of shape (num_frames, 3, 3) - rotation matrices
        trajectory_2d: numpy array of shape (num_frames, 2) - XY positions for plotting
    """
    # Convert to numpy
    if isinstance(poses, torch.Tensor):
        poses = poses.cpu().numpy()
    
    # Remove batch dimension if present: (1, N, 7) -> (N, 7)
    if len(poses.shape) == 3 and poses.shape[0] == 1:
        poses = poses[0]
    
    # Now poses should be (num_frames, 7)
    assert len(poses.shape) == 2 and poses.shape[1] == 7, f"Expected shape (N, 7), got {poses.shape}"
    
    # Extract quaternion and translation
    translations = poses[:, :3]   # [tx, ty, tz] - first 3 values
    quaternions = poses[:, 3:]    # [qx, qy, qz, qw] - last 4 values
    
    # Convert from OpenCV convention (X=right, Y=down, Z=forward) 
    # to top-down convention (X=right, Y=forward, Z=up)
    translations = translations[:, [0, 2, 1]]  # Reorder to [tx, tz, ty]

    # Convert quaternions to rotation matrices
    orientations = quaternion_to_rotation_matrix(quaternions)
    
    # VIPE predicts real metric values directly
    trajectory = translations
    trajectory_2d = trajectory[:, :2]  # XY positions
    
    return trajectory, orientations, trajectory_2d

def process_vipe_segments_for_trajectories(segments_dir, config_path=None, pipeline_config="default"):
    """
    Process all video segments using VIPE SLAM to extract trajectories.
    
    Args:
        segments_dir: Path to segments directory containing videos
        config_path: Path to VIPE configuration file (optional)
        pipeline_config: Name of pipeline config to use (default: "default")
        
    Returns:
        all_trajectories: List of trajectory arrays for each segment
        all_orientations: List of orientation arrays for each segment
        all_trajectories_2d: List of 2D trajectory arrays for plotting
        segment_info: List of dictionaries containing segment metadata
    """
    print("Processing video segments with VIPE SLAM for trajectory extraction...")
    
    # Find all video files or frame directories
    video_files = glob.glob(os.path.join(segments_dir, "*.mp4"))
    video_files += glob.glob(os.path.join(segments_dir, "*.MP4"))
    
    if not video_files:
        print(f"No video files found in {segments_dir}")
        return [], [], [], []
    
    print(f"Found {len(video_files)} video files to process")
    
    all_trajectories = []
    all_orientations = []
    all_trajectories_2d = []
    segment_info = []
    
    # Load VIPE configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config = OmegaConf.create(config)
    else:
        # Use default config
        config = OmegaConf.create({
            'pipeline': pipeline_config,
            'streams': {'type': 'raw_mp4_stream'}
        })
    
    for video_path in video_files:
        segment_name = os.path.basename(video_path).replace('.mp4', '').replace('.MP4', '')
        print(f"\nProcessing segment: {segment_name}")
        
        try:
            # Create video stream from file
            from vipe.streams.raw_mp4_stream import RawMp4Stream
            video_stream = RawMp4Stream(Path(video_path))
            
            # Create and run SLAM pipeline
            from vipe.pipeline.default import DefaultAnnotationPipeline
            
            # Configure pipeline with full SLAM configuration
            pipeline_cfg = OmegaConf.create({
                'init': {
                    'camera_type': 'pinhole',
                    'instance': None
                },
                'slam': {
                    'buffer': 1024,
                    'beta': 0.3,
                    'filter_thresh': 1.0,  # Enable motion filtering for robustness
                    'warmup': 8,  # More frames for stable initialization
                    'keyframe_thresh': 4.0,  # Higher threshold = fewer keyframes, more stable
                    'frontend_thresh': 12.0,  # More lenient threshold for sampled frames
                    'frontend_window': 20,
                    'frontend_radius': 3,  # Larger radius for better connectivity
                    'frontend_nms': 1,
                    'seq_init': True,
                    'frontend_backend_iters': [8, 16, 32],  # Fewer iterations, more stable
                    'backend_thresh': 16.0,  # More lenient for sampled frames
                    'backend_radius': 3,
                    'backend_nms': 2,
                    'backend_iters': 12,
                    'init_disp': 4.0,  # Lower initial disparity for better convergence
                    'optimize_intrinsics': False,
                    'optimize_rig_rotation': False,
                    'cross_view': False,  # Disable cross-view for simpler optimization
                    'cross_view_idx': None,
                    'adaptive_cross_view': False,
                    'infill_chunk_size': 16,
                    'infill_dense_disp': True,  # Enable dense disparity infill
                    'map_filter_thresh': 0.1,  # Higher filtering for cleaner map
                    'visualize': False,
                    'keyframe_depth': 'metric3d-small',
                    'n_views': 1,
                    'height': 384,
                    'width': 512,
                    'sparse_tracks': {'name': 'dummy'},
                    'ba': {
                        'dense_disp_alpha': 0.01  # Increased regularization for stability
                    }
                },
                'post': {
                    'depth_align_model': None
                },
                'output': {
                    'path': '/tmp/vipe_output',
                    'save_artifacts': False,
                    'save_viz': False,
                    'save_slam_map': False
                }
            })
            
            pipeline = DefaultAnnotationPipeline(
                init=pipeline_cfg.init,
                slam=pipeline_cfg.slam,
                post=pipeline_cfg.post,
                output=pipeline_cfg.output
            )
            
            # Run pipeline and get SLAM output
            pipeline.return_payload = True
            result = pipeline.run(video_stream)
            slam_output = result.payload
            
            # Extract trajectory from SLAM output
            if slam_output is None:
                print(f"Warning: No SLAM output for {segment_name}")
                continue
            
            # Get poses from trajectory (shape: [num_frames, 7] for quaternion + translation)
            poses = slam_output.trajectory.data
            
            # Extract number of poses from tensor shape
            num_poses = poses.shape[0]
            if num_poses == 0:
                print(f"Warning: No poses extracted for {segment_name}")
                continue
            
            print(f"Extracted {num_poses} poses from SLAM")
            
            # Extract trajectory (VIPE predicts real metric values)
            trajectory, orientations, trajectory_2d = extract_trajectory_from_poses(poses)
            
            all_trajectories.append(trajectory)
            all_orientations.append(orientations)
            all_trajectories_2d.append(trajectory_2d)
            
            # Store segment information
            trajectory_length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
            
            info = {
                'segment_name': segment_name,
                'num_frames': len(trajectory),
                'trajectory_length_meters': trajectory_length
            }
            segment_info.append(info)
            
            print(f"Extracted trajectory with {len(trajectory)} camera positions")
            print(f"Trajectory length: {trajectory_length:.2f} meters")
            
        except Exception as e:
            print(f"Error processing {segment_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_trajectories, all_orientations, all_trajectories_2d, segment_info

def load_ground_truth_trajectory(segment_path, segment_name):
    """
    Load ground truth trajectory from CSV file.
    
    Args:
        segment_path: Path to the segment directory
        segment_name: Name of the segment
        
    Returns:
        gt_trajectory_2d: numpy array of shape (N, 2) containing GT trajectory in XY plane
        None if file not found or error loading
    """
    try:
        # Construct path to GT trajectory CSV file
        gt_csv_path = os.path.join(segment_path, f"{segment_name}_trajectory.csv")
        
        if not os.path.exists(gt_csv_path):
            print(f"Warning: Ground truth trajectory file not found: {gt_csv_path}")
            return None
        
        # Load CSV file
        gt_df = pd.read_csv(gt_csv_path)
        
        # Check if required columns exist
        if 'x' not in gt_df.columns or 'y' not in gt_df.columns:
            print(f"Warning: GT trajectory CSV missing x/y columns: {gt_csv_path}")
            return None
        
        # Extract x, y coordinates
        gt_trajectory_2d = gt_df[['x', 'y']].values
        
        # Remove any rows with NaN values
        gt_trajectory_2d = gt_trajectory_2d[~np.isnan(gt_trajectory_2d).any(axis=1)]
        
        print(f"✓ Loaded GT trajectory with {len(gt_trajectory_2d)} points from {os.path.basename(gt_csv_path)}")
        return gt_trajectory_2d
        
    except Exception as e:
        print(f"Error loading GT trajectory from {gt_csv_path}: {e}")
        return None

def save_trajectory_data(trajectory, orientations, output_path, frame_info=None):
    """
    Save trajectory data to various formats.
    
    Args:
        trajectory: numpy array of camera positions
        orientations: numpy array of rotation matrices
        output_path: base path for output files
        frame_info: optional dictionary containing frame metadata
    """
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as numpy arrays
    np.save(f"{output_path}_trajectory.npy", trajectory)
    np.save(f"{output_path}_orientations.npy", orientations)
    
    # Save as CSV for easy viewing
    trajectory_df = pd.DataFrame(trajectory, columns=['x', 'y', 'z'])
    trajectory_df['frame_id'] = range(len(trajectory))
    
    # Add metadata to CSV
    if frame_info:
        for key, value in frame_info.items():
            if isinstance(value, (int, float, str, bool)):
                trajectory_df[key] = value
    
    trajectory_df.to_csv(f"{output_path}_trajectory.csv", index=False)
    
    # Save as JSON for metadata
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native_type(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            # Handle NaN values
            if np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native_type(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native_type(item) for item in obj]
        return obj
    
    trajectory_data = {
        'trajectory': convert_to_native_type(trajectory),
        'orientations': convert_to_native_type(orientations),
        'metadata': convert_to_native_type(frame_info) if frame_info else {}
    }
    
    with open(f"{output_path}_trajectory.json", 'w') as f:
        json.dump(trajectory_data, f, indent=2)
    
    print(f"✓ Saved trajectory data to {output_path}_trajectory.*")

def visualize_trajectory_with_gt(trajectory, gt_trajectory_2d=None, output_path=None, title="Camera Trajectory"):
    """
    Create visualization of trajectory with optional ground truth comparison.
    
    Args:
        trajectory: numpy array of camera positions (3D)
        gt_trajectory_2d: optional numpy array of ground truth trajectory (2D)
        output_path: optional path to save the plot
        title: title for the plot
    """
    try:
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(16, 8))
        
        # 3D view
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b.-', markersize=4, linewidth=2)
        ax1.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], c='green', s=100, label='Start')
        ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], c='red', s=100, label='End')
        ax1.set_xlabel('X (meters)')
        ax1.set_ylabel('Y (meters)')
        ax1.set_zlabel('Z (meters)')
        ax1.set_title(f'3D View - {title}')
        ax1.legend()
        
        # Top-down view (X-Y plane)
        ax2 = fig.add_subplot(132)
        ax2.plot(trajectory[:, 0], trajectory[:, 1], 'b.-', markersize=6, linewidth=2, label='VIPE SLAM')
        ax2.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, label='Start')
        ax2.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, label='End')
        
        # Add ground truth if available
        if gt_trajectory_2d is not None:
            ax2.plot(gt_trajectory_2d[:, 0], gt_trajectory_2d[:, 1], 'r.-', markersize=4, linewidth=2, 
                    alpha=0.7, label='Ground Truth')
            ax2.scatter(gt_trajectory_2d[0, 0], gt_trajectory_2d[0, 1], c='darkgreen', s=80, 
                       marker='s', label='GT Start')
            ax2.scatter(gt_trajectory_2d[-1, 0], gt_trajectory_2d[-1, 1], c='darkred', s=80, 
                       marker='s', label='GT End')
        
        ax2.set_xlabel('X (meters)')
        ax2.set_ylabel('Y (meters)')
        ax2.set_title(f'Top-Down View - {title}')
        ax2.legend()
        ax2.axis('equal')
        ax2.grid(True, alpha=0.3)
        
        # Comparison plot
        ax3 = fig.add_subplot(133)
        if gt_trajectory_2d is not None:
            pred_2d = trajectory[:, :2]
            
            ax3.plot(pred_2d[:, 0], pred_2d[:, 1], 'b-', linewidth=3, alpha=0.8, label='VIPE SLAM')
            ax3.plot(gt_trajectory_2d[:, 0], gt_trajectory_2d[:, 1], 'r--', linewidth=2, alpha=0.8, label='Ground Truth')
            
            # Mark start/end points
            ax3.scatter(pred_2d[0, 0], pred_2d[0, 1], c='blue', s=100, marker='o')
            ax3.scatter(pred_2d[-1, 0], pred_2d[-1, 1], c='blue', s=100, marker='s')
            ax3.scatter(gt_trajectory_2d[0, 0], gt_trajectory_2d[0, 1], c='red', s=80, marker='o')
            ax3.scatter(gt_trajectory_2d[-1, 0], gt_trajectory_2d[-1, 1], c='red', s=80, marker='s')
            
            # Calculate RMSE if trajectories have similar lengths
            if abs(len(pred_2d) - len(gt_trajectory_2d)) <= 5:
                min_len = min(len(pred_2d), len(gt_trajectory_2d))
                rmse = np.sqrt(np.mean(np.sum((pred_2d[:min_len] - gt_trajectory_2d[:min_len])**2, axis=1)))
                ax3.text(0.02, 0.98, f'RMSE: {rmse:.3f}m', transform=ax3.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax3.set_title('Trajectory Comparison')
        else:
            pred_2d = trajectory[:, :2]
            ax3.plot(pred_2d[:, 0], pred_2d[:, 1], 'b.-', markersize=6, linewidth=2)
            ax3.scatter(pred_2d[0, 0], pred_2d[0, 1], c='green', s=100)
            ax3.scatter(pred_2d[-1, 0], pred_2d[-1, 1], c='red', s=100)
            ax3.set_title('VIPE SLAM Trajectory')
        
        ax3.set_xlabel('X (meters)')
        ax3.set_ylabel('Y (meters)')
        ax3.legend()
        ax3.axis('equal')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(f"{output_path}_trajectory_plot.png", dpi=300, bbox_inches='tight')
            print(f"✓ Saved trajectory plot to {output_path}_trajectory_plot.png")
        
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")

def create_trajectory_summary_report(segment_info, output_path=None):
    """
    Create a summary report of all trajectory statistics.
    
    Args:
        segment_info: List of segment metadata dictionaries
        output_path: optional path to save the report
    """
    if not segment_info:
        print("No segment information available for report.")
        return
    
    # Calculate summary statistics
    trajectory_lengths = [info['trajectory_length_meters'] for info in segment_info]
    
    report = f"""
=== VIPE SLAM Trajectory Analysis Report ===

Total Segments Processed: {len(segment_info)}

Trajectory Length Statistics (meters):
  Mean: {np.mean(trajectory_lengths):.2f}
  Median: {np.median(trajectory_lengths):.2f}
  Std Dev: {np.std(trajectory_lengths):.2f}
  Min: {np.min(trajectory_lengths):.2f}
  Max: {np.max(trajectory_lengths):.2f}

Detailed Segment Information:
"""
    
    for i, info in enumerate(segment_info, 1):
        report += f"""
{i:2d}. Segment: {info['segment_name']}
    Trajectory Length: {info['trajectory_length_meters']:.2f}m
    Number of Frames: {info['num_frames']}
"""
    
    print(report)
    
    if output_path:
        with open(f"{output_path}_trajectory_report.txt", 'w') as f:
            f.write(report)
        print(f"✓ Saved trajectory report to {output_path}_trajectory_report.txt")

def main():
    """Main function to extract trajectories from VIPE SLAM output."""
    print("=== Extracting Trajectories from VIPE SLAM ===")
    
    # Configuration
    segments_dir = "/home/hongyuan/world-decoder/dataset/video_samples"  # Change this to your video directory
    output_dir = "/home/hongyuan/world-decoder/world-decoder_code_vipe/output/trajectories"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing videos from: {segments_dir}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Process all video segments
        print("\n--- Processing All Video Segments with VIPE SLAM ---")
        all_trajectories, all_orientations, all_trajectories_2d, segment_info = process_vipe_segments_for_trajectories(
            segments_dir=segments_dir
        )
        
        if not all_trajectories:
            print("No trajectories were successfully extracted!")
            return
        
        print(f"\n✓ Successfully processed {len(all_trajectories)} segments")
        
        # Save individual trajectory data
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        for i, (trajectory, orientations, info) in enumerate(zip(all_trajectories, all_orientations, segment_info)):
            segment_output_path = os.path.join(output_dir, f"{info['segment_name']}_{timestamp}")
            save_trajectory_data(trajectory, orientations, segment_output_path, info)
            
            # Visualize individual trajectory
            visualize_trajectory_with_gt(
                trajectory,
                None,
                segment_output_path,
                f"VIPE SLAM: {info['segment_name']}"
            )
        
        # Create summary report
        report_output_path = os.path.join(output_dir, f"summary_{timestamp}")
        create_trajectory_summary_report(segment_info, report_output_path)
        
        print(f"\n=== Processing Complete ===")
        print(f"✓ Processed {len(all_trajectories)} segments")
        print(f"✓ Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

