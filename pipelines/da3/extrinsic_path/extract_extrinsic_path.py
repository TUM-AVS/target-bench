"""
Trajectory extraction helpers for Depth Anything 3 extrinsics.

DA3 returns camera extrinsics in OpenCV/Colmap world-to-camera convention. The
trajectory conversion below follows the same coordinate mapping used by the
VGGT evaluator so DA3 scores remain comparable inside target-bench.
"""

import os

import numpy as np
import pandas as pd
import torch


def extract_trajectory_from_extrinsics(extrinsic_matrices, scale_factor=1.0):
    """
    Extract camera trajectory from world-to-camera extrinsic matrices.

    Args:
        extrinsic_matrices: torch.Tensor or numpy array of shape
            (batch, num_cameras, 3/4, 4) or (num_cameras, 3/4, 4).
        scale_factor: Metric scale factor applied to camera centers.

    Returns:
        trajectory: numpy array of shape (num_cameras - 1, 3)
        orientations: numpy array of shape (num_cameras, 3, 3)
        trajectory_2d: numpy array of shape (num_cameras - 1, 2)
    """
    if isinstance(extrinsic_matrices, torch.Tensor):
        extrinsic_matrices = extrinsic_matrices.detach().cpu().numpy()

    extrinsic_matrices = np.asarray(extrinsic_matrices, dtype=np.float32)

    if extrinsic_matrices.ndim == 4:
        extrinsic_matrices = extrinsic_matrices[0]

    if extrinsic_matrices.ndim != 3 or extrinsic_matrices.shape[1] < 3 or extrinsic_matrices.shape[2] < 4:
        raise ValueError(f"Unexpected extrinsics shape: {extrinsic_matrices.shape}")

    num_cameras = extrinsic_matrices.shape[0]
    if num_cameras < 2:
        raise ValueError("At least two camera poses are required to extract a trajectory.")

    trajectory = np.zeros((num_cameras - 1, 3), dtype=np.float32)

    r_w2c = extrinsic_matrices[:, :3, :3]
    t_w2c = extrinsic_matrices[:, :3, 3]

    r_c2w = np.transpose(r_w2c, (0, 2, 1))
    t_c2w = -np.matmul(r_c2w, t_w2c[:, :, None])[:, :, 0]

    trajectory_raw = t_c2w * scale_factor
    orientations = r_c2w

    # Keep the target-bench/VGGT coordinate convention for comparable metrics.
    trajectory[:, 0] = trajectory_raw[:-1, 0]
    trajectory[:, 1] = trajectory_raw[:-1, 2]
    trajectory[:, 2] = trajectory_raw[:-1, 1]

    trajectory_2d = trajectory[:, :2]

    return trajectory, orientations, trajectory_2d


def load_ground_truth_trajectory(segment_path, segment_name):
    """
    Load ground-truth 2D trajectory from a segment CSV file.
    """
    gt_csv_path = os.path.join(segment_path, f"{segment_name}_trajectory.csv")

    try:
        if not os.path.exists(gt_csv_path):
            print(f"Warning: Ground truth trajectory file not found: {gt_csv_path}")
            return None

        gt_df = pd.read_csv(gt_csv_path)

        if "x" not in gt_df.columns or "y" not in gt_df.columns:
            print(f"Warning: GT trajectory CSV missing x/y columns: {gt_csv_path}")
            return None

        gt_trajectory_2d = gt_df[["x", "y"]].values
        gt_trajectory_2d = gt_trajectory_2d[~np.isnan(gt_trajectory_2d).any(axis=1)]

        print(f"✓ Loaded GT trajectory with {len(gt_trajectory_2d)} points from {os.path.basename(gt_csv_path)}")
        return gt_trajectory_2d

    except Exception as exc:
        print(f"Error loading GT trajectory from {gt_csv_path}: {exc}")
        return None
