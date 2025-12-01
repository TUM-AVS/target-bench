"""
Extract extrinsic path and build trajectories from EgoWalk segments using SpaTrackerV2.

This module adapts the extract_extrinsic_path.py functionality to work with SpaTrackerV2
instead of VGGT for pose estimation and trajectory extraction.

Key functionality:
1. Processes EgoWalk segments using SpaTrackerV2 models from compute_scale_factor.py
2. Extracts camera trajectories from pose predictions with real metric scale factors
3. Projects trajectories onto the X-Y plane for visualization
4. Creates comprehensive visualizations and reports with tracking information
5. Handles multiple segments and provides statistical analysis

Main functions:
- process_egowalk_segments_for_trajectories(): Process all segments and extract trajectories
- extract_trajectory_from_extrinsics(): Convert extrinsic matrices to trajectory points
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

# Add the scale_factor module to path
sys.path.append('/home/hongyuan/world-decoder/world-decoder_code_spatracker')
sys.path.append('/home/hongyuan/world-decoder/SpaTrackerV2')

from scale_factor.compute_scale_factor import process_egowalk_segments, process_single_segment, load_config

def extract_trajectory_from_extrinsics(extrinsic_matrices, scale_factor=1.0):
    """
    Extract camera trajectory from extrinsic matrices from SpaTrackerV2.
    
    Args:
        extrinsic_matrices: numpy array of shape (num_frames, 3, 4) or (num_frames, 4, 4)
                           Extrinsic matrices from SpaTrackerV2 models
        scale_factor: float, scale factor to apply to positions
        
    Returns:
        trajectory: numpy array of shape (num_frames-1, 3) containing camera positions
        orientations: numpy array of shape (num_frames, 3, 3) containing rotation matrices
        trajectory_2d: numpy array of shape (num_frames-1, 2) containing XY positions for plotting
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(extrinsic_matrices, torch.Tensor):
        extrinsic_matrices = extrinsic_matrices.cpu().numpy()
    
    # Handle different matrix formats
    if len(extrinsic_matrices.shape) == 3:
        if extrinsic_matrices.shape[-1] == 4 and extrinsic_matrices.shape[-2] == 4:
            # Full 4x4 transformation matrices - extract 3x4 part
            extrinsic_matrices = extrinsic_matrices[:, :3, :]
        elif extrinsic_matrices.shape[-1] == 4 and extrinsic_matrices.shape[-2] == 3:
            # Already 3x4 matrices
            pass
        else:
            raise ValueError(f"Unexpected extrinsic matrix shape: {extrinsic_matrices.shape}")
    else:
        raise ValueError(f"Expected 3D array, got shape: {extrinsic_matrices.shape}")
    
    num_frames = extrinsic_matrices.shape[0]
    trajectory = np.zeros((num_frames - 1, 3))  # Remove the last frame
    orientations = np.zeros((num_frames, 3, 3))

    # Extract rotation and translation from extrinsic matrices
    R_w2c = extrinsic_matrices[:, :3, :3]  # World to camera rotation (num_frames, 3, 3)
    t_w2c = extrinsic_matrices[:, :3, 3]   # World to camera translation (num_frames, 3)

    # Convert to camera-to-world transformation
    R_c2w = np.transpose(R_w2c, (0, 2, 1))  # Camera to world rotation
    t_c2w = -np.matmul(R_c2w, t_w2c[:, :, None])[:, :, 0]  # Camera positions in world coordinates

    # Apply scale factor to get real metric distances
    trajectory_raw = t_c2w * scale_factor
    orientations = R_c2w
    
    # Apply coordinate transformation to match ground truth coordinate system
    # SpaTrackerV2 output coordinates transformation
    trajectory[:, 0] = trajectory_raw[:-1, 0]   # x stays x
    trajectory[:, 1] = -trajectory_raw[:-1, 2]   # y becomes z  
    trajectory[:, 2] = trajectory_raw[:-1, 1]   # z becomes y
    
    # Extract 2D trajectory for plotting (X-Y plane)
    trajectory_2d = trajectory[:, :2]  # Only X and Y coordinates
    
    return trajectory, orientations, trajectory_2d

def extract_trajectory_from_c2w(extrinsic_matrices, scale_factor, current_pos, move_to_origin=False, reorth=True):
    """
    Extract camera trajectory from extrinsic matrices.
    
    Simple approach:
    1. Remove the first frame
    2. Use the second frame (index 1) as origin
    3. Extract translations relative to origin
    
    Args:
        extrinsic_matrices: array of shape (num_frames, 4, 4)
        scale_factor: float, scale factor to apply to positions
        current_pos: not used in simplified version
        move_to_origin: not used in simplified version  
        reorth: not used in simplified version
        
    Returns:
        trajectory: array of shape (num_frames-1, 3) containing camera positions
        orientations: array of shape (num_frames-1, 3, 3) containing rotation matrices
        trajectory_2d: array of shape (num_frames-1, 2) containing XY positions
    """
    
    # Convert to torch if needed
    if isinstance(extrinsic_matrices, np.ndarray):
        extrinsic_matrices = torch.from_numpy(extrinsic_matrices).float()
    
    # Remove first frame, work with frames [1:]
    matrices = extrinsic_matrices[1:]
    
    # Use first remaining frame (original index 1) as origin
    origin = matrices[0]
    origin_inv = torch.linalg.inv(origin)
    
    # Transform all matrices relative to origin
    relative_matrices = origin_inv.unsqueeze(0) @ matrices
    
    # Extract translations and apply scale factor
    translations = relative_matrices[:, :3, 3] * scale_factor
    
    # Extract rotations
    orientations = relative_matrices[:, :3, :3]
    
    # Apply coordinate transformation: (x, y, z) -> (x, z, y)
    trajectory = torch.zeros_like(translations)
    trajectory[:, 0] = translations[:, 0]  # x stays x
    trajectory[:, 1] = translations[:, 2]  # y becomes z
    trajectory[:, 2] = translations[:, 1]  # z becomes y
    
    # Extract 2D trajectory for plotting (X-Y plane)
    trajectory_2d = trajectory[:, :2]
    
    return trajectory, orientations, trajectory_2d

def process_egowalk_segments_for_trajectories(segments_dir="/home/hongyuan/world-decoder/dataset/EgoWalk_samples", 
                                            config_path="configs/spatracker_pcd_configs.yml"):
    """
    Process all EgoWalk segments to extract trajectories using SpaTrackerV2.
    
    Args:
        segments_dir: Path to EgoWalk segments directory
        config_path: Path to configuration file
        
    Returns:
        all_trajectories: List of trajectory arrays for each segment
        all_orientations: List of orientation arrays for each segment
        all_trajectories_2d: List of 2D trajectory arrays for plotting
        segment_info: List of dictionaries containing segment metadata
    """
    print("Processing EgoWalk segments for trajectory extraction using SpaTrackerV2...")

    with open("world-decoder_code_spatracker/video_processing_results_frame_based.json", 'r') as f:
        video_processing_results = json.load(f)

    video_processing_data = {}
    for result in video_processing_results:
        folder_name = result['folder_name']
        video_processing_data[folder_name] = result
    
    # Get results from process_egowalk_segments (now memory optimized)
    results = process_egowalk_segments(config_path, segments_dir, video_processing_data)
    
    if not results:
        print("No segments were successfully processed!")
        return [], [], [], []
    
    all_trajectories = []
    all_orientations = []
    all_trajectories_2d = []
    segment_info = []
    
    for result in results:
        segment_name = result['segment_name']
        scale_factor = result['scale_factor']
        extrinsic_matrices = result['extrinsic']
        current_pos = result['current_pos']
        
        print(f"\nExtracting trajectory for segment: {segment_name}")
        print(f"Scale factor: {scale_factor:.4f}")
        
        if scale_factor == float('inf') or scale_factor <= 0:
            print(f"Skipping segment {segment_name} due to invalid scale factor")
            continue
        
        # Extract trajectory from extrinsic matrices
        # trajectory, orientations, trajectory_2d = extract_trajectory_from_extrinsics(
        #     extrinsic_matrices, scale_factor
        # )
        trajectory, orientations, trajectory_2d = extract_trajectory_from_c2w(
            extrinsic_matrices, scale_factor, current_pos
        )
        
        all_trajectories.append(trajectory)
        all_orientations.append(orientations)
        all_trajectories_2d.append(trajectory_2d)
        
        # Store segment information
        info = {
            'segment_name': segment_name,
            'scale_factor': scale_factor,
            'real_displacement': result['real_displacement'],
            'camera_displacement': result['camera_displacement'],
            'num_cameras': len(trajectory),
            'trajectory_length_meters': np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)) if len(trajectory) > 1 else 0.0
        }
        segment_info.append(info)
        
        print(f"Extracted trajectory with {len(trajectory)} camera positions")
        print(f"Trajectory length: {info['trajectory_length_meters']:.2f} meters")
    
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

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

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
    trajectory_df['camera_id'] = range(len(trajectory))
    
    # Add metadata to CSV only if it's scalar or can be broadcast to match trajectory length
    if frame_info:
        for key, value in frame_info.items():
            try:
                # Convert numpy types
                if isinstance(value, (np.ndarray, np.float32, np.float64, np.int32, np.int64, np.bool_)):
                    converted_value = convert_numpy_types(value)
                else:
                    converted_value = value
                
                # Only add to DataFrame if it's a scalar or has compatible length
                if isinstance(converted_value, (int, float, str, bool)):
                    # Scalar values can be broadcast to all rows
                    trajectory_df[key] = converted_value
                elif isinstance(converted_value, list) and len(converted_value) == len(trajectory):
                    # List with matching length
                    trajectory_df[key] = converted_value
                else:
                    # Skip values that don't match
                    print(f"Skipping metadata field '{key}' in CSV (length mismatch or complex type)")
                    
            except Exception as e:
                print(f"Warning: Could not add metadata field '{key}' to CSV: {e}")
                continue
                
    trajectory_df.to_csv(f"{output_path}_trajectory.csv", index=False)
    
    # Save as JSON for metadata - convert all numpy types first
    trajectory_data = {
        'trajectory': trajectory.tolist(),
        'orientations': orientations.tolist(),
        'metadata': convert_numpy_types(frame_info) if frame_info else {}
    }
    
    with open(f"{output_path}_trajectory.json", 'w') as f:
        json.dump(trajectory_data, f, indent=2)
    
    print(f"✓ Saved trajectory data to {output_path}_trajectory.*")

def visualize_trajectory_with_gt(trajectory, gt_trajectory_2d=None, output_path=None, title="Camera Trajectory"):
    """
    Create visualization of trajectory with optional ground truth comparison.
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        
        # Extract 2D trajectory for plotting
        pred_2d = trajectory[:, :2]  # Extract X-Y from predicted
        
        # Plot SpaTrackerV2 trajectory
        ax.plot(pred_2d[:, 0], pred_2d[:, 1], 'b-', linewidth=3, alpha=0.8, label='SpaTrackerV2')
        ax.scatter(pred_2d[0, 0], pred_2d[0, 1], c='blue', s=100, marker='o', label='Start')
        ax.scatter(pred_2d[-1, 0], pred_2d[-1, 1], c='blue', s=100, marker='s', label='End')
        
        # Add ground truth trajectory if available
        if gt_trajectory_2d is not None:
            ax.plot(gt_trajectory_2d[:, 0], gt_trajectory_2d[:, 1], 'r--', linewidth=2, alpha=0.8, label='Ground Truth')
            ax.scatter(gt_trajectory_2d[0, 0], gt_trajectory_2d[0, 1], c='red', s=80, marker='o', label='GT Start')
            ax.scatter(gt_trajectory_2d[-1, 0], gt_trajectory_2d[-1, 1], c='red', s=80, marker='s', label='GT End')
            
            # Calculate alignment metrics if both trajectories have similar lengths
            if abs(len(pred_2d) - len(gt_trajectory_2d)) <= 5:
                min_len = min(len(pred_2d), len(gt_trajectory_2d))
                pred_trimmed = pred_2d[:min_len]
                gt_trimmed = gt_trajectory_2d[:min_len]
                
                # Calculate RMSE
                rmse = np.sqrt(np.mean(np.sum((pred_trimmed - gt_trimmed)**2, axis=1)))
                ax.text(0.02, 0.98, f'RMSE: {rmse:.3f}m', transform=ax.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plot_title = f'SpaTrackerV2 vs Ground Truth'
        else:
            plot_title = f'{title} - SpaTrackerV2 Trajectory'
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title(plot_title)
        ax.legend()
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(f"{output_path}_trajectory_plot.png", dpi=300, bbox_inches='tight')
            print(f"✓ Saved trajectory comparison plot to {output_path}_trajectory_plot.png")
        
        # Close the plot instead of showing it to avoid blocking
        plt.close(fig)
        
    except ImportError:
        print("Matplotlib not available for visualization")
    except Exception as e:
        print(f"Error creating visualization: {e}")
        # Make sure to close any open figures
        plt.close('all')

def create_trajectory_summary_report(segment_info, output_path=None):
    """
    Create a summary report of all trajectory statistics.
    """
    if not segment_info:
        print("No segment information available for report.")
        return
    
    # Calculate summary statistics
    scale_factors = [info['scale_factor'] for info in segment_info]
    trajectory_lengths = [info['trajectory_length_meters'] for info in segment_info]
    real_displacements = [info['real_displacement'] for info in segment_info]
    
    report = f"""
=== EgoWalk Segments Trajectory Analysis Report (SpaTrackerV2) ===

Total Segments Processed: {len(segment_info)}

Scale Factor Statistics:
  Mean: {np.mean(scale_factors):.4f}
  Median: {np.median(scale_factors):.4f}
  Std Dev: {np.std(scale_factors):.4f}
  Min: {np.min(scale_factors):.4f}
  Max: {np.max(scale_factors):.4f}

Trajectory Length Statistics (meters):
  Mean: {np.mean(trajectory_lengths):.2f}
  Median: {np.median(trajectory_lengths):.2f}
  Std Dev: {np.std(trajectory_lengths):.2f}
  Min: {np.min(trajectory_lengths):.2f}
  Max: {np.max(trajectory_lengths):.2f}

Real Displacement Statistics (meters):
  Mean: {np.mean(real_displacements):.2f}
  Median: {np.median(real_displacements):.2f}
  Std Dev: {np.std(real_displacements):.2f}
  Min: {np.min(real_displacements):.2f}
  Max: {np.max(real_displacements):.2f}

Detailed Segment Information:
"""
    
    for i, info in enumerate(segment_info, 1):
        report += f"""
{i:2d}. Segment: {info['segment_name']}
    Scale Factor: {info['scale_factor']:.4f}
    Real Displacement: {info['real_displacement']:.2f}m
    Trajectory Length: {info['trajectory_length_meters']:.2f}m
    Number of Cameras: {info['num_cameras']}
"""
    
    print(report)
    
    if output_path:
        with open(f"{output_path}_trajectory_report.txt", 'w') as f:
            f.write(report)
        print(f"✓ Saved trajectory report to {output_path}_trajectory_report.txt")

def main():
    """Main function to extract extrinsic paths and build trajectories from EgoWalk segments using SpaTrackerV2."""
    print("=== Extracting Extrinsic Paths and Building Trajectories using SpaTrackerV2 ===")
    
    # Configuration
    config_path = "configs/spatracker_pcd_configs.yml"
    segments_dir = "/home/hongyuan/world-decoder/dataset/Benchmark"
    output_dir = "/home/hongyuan/world-decoder/world-decoder_code_spatracker/output_temp/trajectories"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing EgoWalk segments from: {segments_dir}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Process all EgoWalk segments to extract trajectories
        print("\n--- Processing All EgoWalk Segments with SpaTrackerV2 ---")
        all_trajectories, all_orientations, all_trajectories_2d, segment_info = process_egowalk_segments_for_trajectories(
            segments_dir=segments_dir,
            config_path=config_path
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
        
        # Create visualizations
        print("\n--- Creating Visualizations ---")
        
        # 1. Visualize individual trajectories with GT comparison
        for i, (trajectory, info) in enumerate(zip(all_trajectories, segment_info)):
            segment_name = info['segment_name']
            print(f"Creating visualization {i+1}/{len(all_trajectories)}: {segment_name}")
            
            try:
                individual_output_path = os.path.join(output_dir, f"{segment_name}_individual")
                
                # Load ground truth trajectory for this segment
                segment_path = os.path.join(segments_dir, segment_name)
                gt_trajectory_2d = load_ground_truth_trajectory(segment_path, segment_name)
                
                # Create visualization with GT comparison
                visualize_trajectory_with_gt(
                    trajectory, 
                    gt_trajectory_2d, 
                    individual_output_path, 
                    f"SpaTrackerV2 - Segment: {segment_name}"
                )
                
            except Exception as e:
                print(f"Error creating visualization for {segment_name}: {e}")
                continue
        
        # 4. Create summary report
        report_output_path = os.path.join(output_dir, f"summary_{timestamp}")
        create_trajectory_summary_report(segment_info, report_output_path)
        
        # 5. Save combined data
        combined_output_path = os.path.join(output_dir, f"combined_trajectories_{timestamp}")
        combined_trajectory = np.concatenate(all_trajectories, axis=0)
        combined_orientations = np.concatenate(all_orientations, axis=0)
        
        combined_metadata = {
            'total_segments': len(all_trajectories),
            'total_cameras': len(combined_trajectory),
            'segment_info': segment_info,
            'processing_timestamp': timestamp,
            'model_used': 'SpaTrackerV2'
        }
        
        save_trajectory_data(combined_trajectory, combined_orientations, combined_output_path, combined_metadata)
        
        print(f"\n=== Processing Complete ===")
        print(f"✓ Processed {len(all_trajectories)} segments using SpaTrackerV2")
        print(f"✓ Generated {len(combined_trajectory)} total camera positions")
        print(f"✓ Results saved to: {output_dir}")
        
        # Print quick statistics
        scale_factors = [info['scale_factor'] for info in segment_info]
        print(f"\nQuick Statistics:")
        print(f"  Mean scale factor: {np.mean(scale_factors):.4f}")
        print(f"  Scale factor range: {np.min(scale_factors):.4f} - {np.max(scale_factors):.4f}")
        
    except Exception as e:
        print(f"Error in EgoWalk segment processing: {str(e)}")
        import traceback
        traceback.print_exc()

def process_single_segment_demo(segment_name="2025_01_17__13_15_23_custom_segment_040_088"):
    """
    Demo function to process a single segment for testing with SpaTrackerV2.
    """
    print(f"=== Demo: Processing Single Segment with SpaTrackerV2 ({segment_name}) ===")
    
    segments_dir = "/home/hongyuan/world-decoder/dataset/EgoWalk_samples"
    segment_path = os.path.join(segments_dir, segment_name)
    output_dir = "/home/hongyuan/world-decoder/world-decoder_code_spatracker/output/trajectories"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(segment_path):
        print(f"Error: Segment path does not exist: {segment_path}")
        return
    
    try:
        # Process single segment
        result = process_single_egowalk_segment(segment_path)
        
        if result:
            # Extract trajectory
            trajectory, orientations, trajectory_2d = extract_trajectory_from_extrinsics(
                result['extrinsic'], result['scale_factor']
            )
            
            print(f"\n✓ Successfully processed segment with SpaTrackerV2: {segment_name}")
            print(f"  Scale factor: {result['scale_factor']:.4f}")
            print(f"  Real displacement: {result['real_displacement']:.2f}m")
            print(f"  Camera positions: {len(trajectory)}")
            
            # Save and visualize
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"{segment_name}_demo_{timestamp}")
            
            save_trajectory_data(trajectory, orientations, output_path, result)
            visualize_trajectory_with_gt(trajectory, None, output_path, f"SpaTrackerV2 Demo: {segment_name}")
            
        else:
            print("Failed to process segment")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
