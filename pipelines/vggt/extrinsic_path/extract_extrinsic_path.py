"""
Extract extrinsic path and build trajectories from EgoWalk segments.

This module has been modified to work with the EgoWalk_samples dataset instead of pkl files.

Key functionality:
1. Processes EgoWalk segments using process_egowalk_segments() from compute_scale_factor.py
2. Extracts camera trajectories from extrinsic matrices with real metric scale factors
3. Projects trajectories onto the X-Y plane for visualization
4. Creates comprehensive visualizations and reports
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
sys.path.append('/home/hongyuan/world-decoder/world-decoder_code_ego')
sys.path.append('/home/hongyuan/world-decoder/vggt_code')

from scale_factor.compute_scale_factor import process_egowalk_segments, load_config
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

def extract_trajectory_from_extrinsics(extrinsic_matrices, scale_factor=1.0):
    """
    Extract camera trajectory from extrinsic matrices.
    
    Args:
        extrinsic_matrices: torch.Tensor or numpy array of shape (batch, num_cameras, 3, 4)
                           Extrinsic matrices in OpenCV convention (camera from world)
        scale_factor: float, scale factor to apply to positions
        
    Returns:
        trajectory: numpy array of shape (num_cameras, 3) containing camera positions
        orientations: numpy array of shape (num_cameras, 3, 3) containing rotation matrices
        trajectory_2d: numpy array of shape (num_cameras, 2) containing XY positions for plotting
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(extrinsic_matrices, torch.Tensor):
        extrinsic_matrices = extrinsic_matrices.cpu().numpy()
    
    # Handle batch dimension - take the first batch
    if len(extrinsic_matrices.shape) == 4:
        extrinsic_matrices = extrinsic_matrices[0]  # Shape: (num_cameras, 3, 4)
    
    num_cameras = extrinsic_matrices.shape[0]
    trajectory = np.zeros((num_cameras - 1, 3)) # remove the first frame (its index is -1)
    orientations = np.zeros((num_cameras, 3, 3))

    # Extract rotation and translation from extrinsic matrices
    R_w2c = extrinsic_matrices[:, :3, :3]  # World to camera rotation (num_cameras, 3, 3)
    t_w2c = extrinsic_matrices[:, :3, 3]   # World to camera translation (num_cameras, 3)

    # Convert to camera-to-world transformation
    R_c2w = np.transpose(R_w2c, (0, 2, 1))  # Camera to world rotation
    t_c2w = -np.matmul(R_c2w, t_w2c[:, :, None])[:, :, 0]  # Camera positions in world coordinates

    # Apply scale factor to get real metric distances
    trajectory_raw = t_c2w * scale_factor
    orientations = R_c2w
    
    # Apply coordinate transformation to match ground truth coordinate system
    # VGGT appears to output coordinates as (x, y, z) but the actual layout is (x, z, y)
    # So we need to reorder: (x, y, z) -> (x, z, y) first, then apply transformation
    # Step 1: Reorder coordinates (x, y, z) -> (x, z, y)
    trajectory[:, 0] = trajectory_raw[:-1, 0]   # x stays x
    trajectory[:, 1] = trajectory_raw[:-1, 2]   # y becomes z (the large value becomes vertical)
    trajectory[:, 2] = trajectory_raw[:-1, 1]   # z becomes y (horizontal movement)
    
    # Extract 2D trajectory for plotting (X-Y plane) - using transformed coordinates
    trajectory_2d = trajectory[:, :2]  # Only X and Y coordinates
    
    return trajectory, orientations, trajectory_2d

def process_egowalk_segments_for_trajectories(segments_dir="/home/hongyuan/world-decoder/dataset/Target_samples", 
                                            config_path="vggt_pcd_configs.yml"):
    """
    Process all EgoWalk segments to extract trajectories.
    
    Args:
        segments_dir: Path to EgoWalk segments directory
        config_path: Path to configuration file
        
    Returns:
        all_trajectories: List of trajectory arrays for each segment
        all_orientations: List of orientation arrays for each segment
        all_trajectories_2d: List of 2D trajectory arrays for plotting
        segment_info: List of dictionaries containing segment metadata
    """
    print("Processing EgoWalk segments for trajectory extraction...")
    
    # Get results from process_egowalk_segments
    results = process_egowalk_segments(config_path, segments_dir)
    
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
        
        print(f"\nExtracting trajectory for segment: {segment_name}")
        print(f"Scale factor: {scale_factor:.4f}")
        
        if scale_factor == float('inf') or scale_factor <= 0:
            print(f"Skipping segment {segment_name} due to invalid scale factor")
            continue
        
        # Extract trajectory from extrinsic matrices
        trajectory, orientations, trajectory_2d = extract_trajectory_from_extrinsics(
            extrinsic_matrices, scale_factor
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
            'trajectory_length_meters': np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
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
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to Python native types
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
                    # Skip values that don't match (like lists with different lengths)
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

# Legacy functions for PKL file processing - kept for reference but not used
# def process_single_frame_trajectory(...) - REMOVED
# def process_multiple_frames_trajectory(...) - REMOVED
# These functions have been replaced by process_egowalk_segments_for_trajectories()

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
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(16, 8))
        
        # Top-down view (X-Y plane) with predicted trajectory
        ax2 = fig.add_subplot(132)
        ax2.plot(trajectory[:, 0], trajectory[:, 1], 'b.-', markersize=6, linewidth=2, label='Predicted')
        ax2.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, label='Start')
        ax2.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, label='End')
        
        # Add ground truth trajectory if available
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
        
        # Comparison plot (if GT available)
        ax3 = fig.add_subplot(133)
        if gt_trajectory_2d is not None:
            # Plot both trajectories for comparison
            pred_2d = trajectory[:, :2]  # Extract X-Y from predicted
            
            ax3.plot(pred_2d[:, 0], pred_2d[:, 1], 'b-', linewidth=3, alpha=0.8, label='Predicted')
            ax3.plot(gt_trajectory_2d[:, 0], gt_trajectory_2d[:, 1], 'r--', linewidth=2, alpha=0.8, label='Ground Truth')
            
            # Mark start/end points
            ax3.scatter(pred_2d[0, 0], pred_2d[0, 1], c='blue', s=100, marker='o', label='Pred Start')
            ax3.scatter(pred_2d[-1, 0], pred_2d[-1, 1], c='blue', s=100, marker='s', label='Pred End')
            ax3.scatter(gt_trajectory_2d[0, 0], gt_trajectory_2d[0, 1], c='red', s=80, marker='o', label='GT Start')
            ax3.scatter(gt_trajectory_2d[-1, 0], gt_trajectory_2d[-1, 1], c='red', s=80, marker='s', label='GT End')
            
            # Calculate alignment metrics if both trajectories have similar lengths
            if abs(len(pred_2d) - len(gt_trajectory_2d)) <= 5:
                min_len = min(len(pred_2d), len(gt_trajectory_2d))
                pred_trimmed = pred_2d[:min_len]
                gt_trimmed = gt_trajectory_2d[:min_len]
                
                # Calculate RMSE
                rmse = np.sqrt(np.mean(np.sum((pred_trimmed - gt_trimmed)**2, axis=1)))
                ax3.text(0.02, 0.98, f'RMSE: {rmse:.3f}m', transform=ax3.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax3.set_title('Trajectory Comparison')
        else:
            # Just plot predicted trajectory
            pred_2d = trajectory[:, :2]
            ax3.plot(pred_2d[:, 0], pred_2d[:, 1], 'b.-', markersize=6, linewidth=2)
            ax3.scatter(pred_2d[0, 0], pred_2d[0, 1], c='green', s=100)
            ax3.scatter(pred_2d[-1, 0], pred_2d[-1, 1], c='red', s=100)
            ax3.set_title('Predicted Trajectory (No GT)')
            
        ax3.set_xlabel('X (meters)')
        ax3.set_ylabel('Y (meters)')
        ax3.legend()
        ax3.axis('equal')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(f"{output_path}_trajectory_plot.png", dpi=300, bbox_inches='tight')
            print(f"✓ Saved trajectory comparison plot to {output_path}_trajectory_plot.png")
        
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")

# Legacy function for backward compatibility
def visualize_trajectory(trajectory, output_path=None, title="Camera Trajectory"):
    """Legacy function - calls the new function without GT."""
    visualize_trajectory_with_gt(trajectory, None, output_path, title)

def visualize_multiple_trajectories(all_trajectories_2d, segment_info, output_path=None, segments_dir="/home/hongyuan/world-decoder/dataset/EgoWalk_samples"):
    """
    Create visualization of multiple trajectories from different segments with GT comparison.
    
    Args:
        all_trajectories_2d: List of 2D trajectory arrays
        segment_info: List of segment metadata dictionaries
        output_path: optional path to save the plot
        segments_dir: Path to segments directory for loading GT trajectories
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        
        # Generate colors for different segments
        colors = cm.tab10(np.linspace(0, 1, len(all_trajectories_2d)))
        
        # Load GT trajectories for all segments
        all_gt_trajectories = []
        for info in segment_info:
            segment_name = info['segment_name']
            segment_path = os.path.join(segments_dir, segment_name)
            gt_traj = load_ground_truth_trajectory(segment_path, segment_name)
            all_gt_trajectories.append(gt_traj)
        
        # Plot 1: All predicted trajectories overlaid
        for i, (traj_2d, info) in enumerate(zip(all_trajectories_2d, segment_info)):
            segment_name = info['segment_name']
            scale_factor = info['scale_factor']
            color = colors[i]
            
            ax1.plot(traj_2d[:, 0], traj_2d[:, 1], '-', 
                    color=color, markersize=4, linewidth=2, 
                    label=f'{segment_name.split("_")[-1]} (SF: {scale_factor:.1f})')
            ax1.scatter(traj_2d[0, 0], traj_2d[0, 1], c='green', s=60, marker='o', zorder=5)
            ax1.scatter(traj_2d[-1, 0], traj_2d[-1, 1], c='red', s=60, marker='s', zorder=5)
        
        ax1.set_xlabel('X (meters)')
        ax1.set_ylabel('Y (meters)')
        ax1.set_title('Predicted Trajectories (All Segments)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Plot 2: All ground truth trajectories overlaid
        for i, (gt_traj, info) in enumerate(zip(all_gt_trajectories, segment_info)):
            if gt_traj is not None:
                segment_name = info['segment_name']
                color = colors[i]
                
                ax2.plot(gt_traj[:, 0], gt_traj[:, 1], '--', 
                        color=color, markersize=4, linewidth=2, 
                        label=f'{segment_name.split("_")[-1]} GT')
                ax2.scatter(gt_traj[0, 0], gt_traj[0, 1], c='green', s=60, marker='o', zorder=5)
                ax2.scatter(gt_traj[-1, 0], gt_traj[-1, 1], c='red', s=60, marker='s', zorder=5)
        
        ax2.set_xlabel('X (meters)')
        ax2.set_ylabel('Y (meters)')
        ax2.set_title('Ground Truth Trajectories (All Segments)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # Plot 3: Overlay comparison of predicted vs GT
        for i, (traj_2d, gt_traj, info) in enumerate(zip(all_trajectories_2d, all_gt_trajectories, segment_info)):
            color = colors[i]
            segment_short = info['segment_name'].split("_")[-1]
            
            # Plot predicted
            ax3.plot(traj_2d[:, 0], traj_2d[:, 1], '-', 
                    color=color, linewidth=2, alpha=0.8, label=f'{segment_short} Pred')
            
            # Plot GT if available
            if gt_traj is not None:
                ax3.plot(gt_traj[:, 0], gt_traj[:, 1], '--', 
                        color=color, linewidth=2, alpha=0.6, label=f'{segment_short} GT')
        
        ax3.set_xlabel('X (meters)')
        ax3.set_ylabel('Y (meters)')
        ax3.set_title('Predicted vs Ground Truth Comparison')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
        
        # Note: The trajectory metrics bar chart has been moved to a separate function
        # for better visualization layout with the GT comparison plots
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(f"{output_path}_multiple_trajectories.png", dpi=300, bbox_inches='tight')
            print(f"✓ Saved multiple trajectories plot to {output_path}_multiple_trajectories.png")
        
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")

def visualize_trajectory_metrics(segment_info, output_path=None):
    """
    Create visualization of trajectory metrics (lengths, scale factors, etc.).
    
    Args:
        segment_info: List of segment metadata dictionaries
        output_path: optional path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Extract data
        segment_names = [info['segment_name'].split('_')[-1] for info in segment_info]
        trajectory_lengths = [info['trajectory_length_meters'] for info in segment_info]
        scale_factors = [info['scale_factor'] for info in segment_info]
        real_displacements = [info['real_displacement'] for info in segment_info]
        
        # Plot 1: Trajectory lengths and scale factors
        ax1_twin = ax1.twinx()
        
        bars1 = ax1.bar(range(len(segment_names)), trajectory_lengths, 
                       alpha=0.7, color='skyblue', label='Trajectory Length (m)')
        bars2 = ax1_twin.bar([x + 0.4 for x in range(len(segment_names))], scale_factors, 
                            width=0.4, alpha=0.7, color='orange', label='Scale Factor')
        
        ax1.set_xlabel('Segment')
        ax1.set_ylabel('Trajectory Length (meters)', color='skyblue')
        ax1_twin.set_ylabel('Scale Factor', color='orange')
        ax1.set_title('Trajectory Metrics by Segment')
        ax1.set_xticks(range(len(segment_names)))
        ax1.set_xticklabels(segment_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            ax1.text(bar1.get_x() + bar1.get_width()/2., height1 + height1*0.01,
                    f'{height1:.1f}m', ha='center', va='bottom', fontsize=8)
            ax1_twin.text(bar2.get_x() + bar2.get_width()/2., height2 + height2*0.01,
                         f'{height2:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Plot 2: Scale factor vs real displacement scatter
        ax2.scatter(real_displacements, scale_factors, c='purple', alpha=0.7, s=100)
        
        # Add segment labels
        for i, (x, y, name) in enumerate(zip(real_displacements, scale_factors, segment_names)):
            ax2.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('Real Displacement (meters)')
        ax2.set_ylabel('Scale Factor')
        ax2.set_title('Scale Factor vs Real Displacement')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(real_displacements, scale_factors, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(real_displacements), max(real_displacements), 100)
        ax2.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        ax2.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(f"{output_path}_trajectory_metrics.png", dpi=300, bbox_inches='tight')
            print(f"✓ Saved trajectory metrics plot to {output_path}_trajectory_metrics.png")
        
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
    scale_factors = [info['scale_factor'] for info in segment_info]
    trajectory_lengths = [info['trajectory_length_meters'] for info in segment_info]
    real_displacements = [info['real_displacement'] for info in segment_info]
    
    report = f"""
=== EgoWalk Segments Trajectory Analysis Report ===

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
    """Main function to extract extrinsic paths and build trajectories from EgoWalk segments."""
    print("=== Extracting Extrinsic Paths and Building Trajectories from EgoWalk Segments ===")
    
    # Configuration
    config_path = "vggt_pcd_configs.yml"
    segments_dir = "/home/hongyuan/world-decoder/dataset/Target_samples"
    output_dir = "/home/hongyuan/world-decoder/world-decoder_code_ego/output/trajectories"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing EgoWalk segments from: {segments_dir}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Process all EgoWalk segments to extract trajectories
        print("\n--- Processing All EgoWalk Segments ---")
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
        
        # 1. Visualize individual trajectories for ALL segments with GT comparison
        for i, (trajectory, info) in enumerate(zip(all_trajectories, segment_info)):
            segment_name = info['segment_name']
            individual_output_path = os.path.join(output_dir, f"{segment_name}_individual")
            
            # Load ground truth trajectory for this segment
            segment_path = os.path.join(segments_dir, segment_name)
            gt_trajectory_2d = load_ground_truth_trajectory(segment_path, segment_name)
            
            # Create visualization with GT comparison
            visualize_trajectory_with_gt(
                trajectory, 
                gt_trajectory_2d, 
                individual_output_path, 
                f"Segment: {segment_name}"
            )
        
        # 2. Visualize all trajectories together with GT comparison
        multiple_output_path = os.path.join(output_dir, f"all_segments_{timestamp}")
        visualize_multiple_trajectories(all_trajectories_2d, segment_info, multiple_output_path, segments_dir)
        
        # 3. Create trajectory metrics visualization
        metrics_output_path = os.path.join(output_dir, f"metrics_{timestamp}")
        visualize_trajectory_metrics(segment_info, metrics_output_path)
        
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
            'processing_timestamp': timestamp
        }
        
        save_trajectory_data(combined_trajectory, combined_orientations, combined_output_path, combined_metadata)
        
        print(f"\n=== Processing Complete ===")
        print(f"✓ Processed {len(all_trajectories)} segments")
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

if __name__ == "__main__":
    main()
