"""
VGGT-Omega trajectory helpers for target-bench evaluation.

VGGT-Omega decodes camera-from-world extrinsics in OpenCV coordinates. The
trajectory conversion intentionally matches the existing VGGT evaluator: the
last image is the appended benchmark first_frame used for scale alignment, so
it is dropped from the predicted trajectory.
"""

import os

import numpy as np
import pandas as pd
import torch


def extract_trajectory_from_extrinsics(extrinsic_matrices, scale_factor=1.0):
    """
    Extract camera trajectory from OpenCV camera-from-world extrinsic matrices.

    Args:
        extrinsic_matrices: torch.Tensor or array with shape (B, N, 3, 4) or
            (N, 3, 4).
        scale_factor: metric scale factor applied to camera centers.

    Returns:
        trajectory: array with shape (N - 1, 3), excluding appended first_frame.
        orientations: array with shape (N, 3, 3), camera-to-world rotations.
        trajectory_2d: array with shape (N - 1, 2), X/Y coordinates.
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
        raise ValueError("At least two extrinsics are required for target-bench trajectory extraction")

    r_w2c = extrinsic_matrices[:, :3, :3]
    t_w2c = extrinsic_matrices[:, :3, 3]
    r_c2w = np.transpose(r_w2c, (0, 2, 1))
    centers = -np.matmul(r_c2w, t_w2c[:, :, None])[:, :, 0]

    trajectory_raw = centers * float(scale_factor)
    trajectory = np.zeros((num_cameras - 1, 3), dtype=np.float32)
    trajectory[:, 0] = trajectory_raw[:-1, 0]
    trajectory[:, 1] = trajectory_raw[:-1, 2]
    trajectory[:, 2] = trajectory_raw[:-1, 1]

    trajectory_2d = trajectory[:, :2]
    return trajectory, r_c2w, trajectory_2d


def load_ground_truth_trajectory(segment_path, segment_name):
    """Load a segment ground-truth trajectory CSV as an (N, 2) array."""
    gt_csv_path = os.path.join(segment_path, f"{segment_name}_trajectory.csv")

    if not os.path.exists(gt_csv_path):
        print(f"Warning: Ground truth trajectory file not found: {gt_csv_path}")
        return None

    try:
        gt_df = pd.read_csv(gt_csv_path)
    except Exception as exc:
        print(f"Error loading GT trajectory from {gt_csv_path}: {exc}")
        return None

    if "x" not in gt_df.columns or "y" not in gt_df.columns:
        print(f"Warning: GT trajectory CSV missing x/y columns: {gt_csv_path}")
        return None

    gt_trajectory_2d = gt_df[["x", "y"]].values
    gt_trajectory_2d = gt_trajectory_2d[~np.isnan(gt_trajectory_2d).any(axis=1)]

    print(f"Loaded GT trajectory with {len(gt_trajectory_2d)} points from {os.path.basename(gt_csv_path)}")
    return gt_trajectory_2d
