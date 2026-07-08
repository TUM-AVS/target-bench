"""Trajectory helpers for VGGT extrinsics."""

import os

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover - torch is required for VGGT inference.
    torch = None


def extract_trajectory_from_extrinsics(extrinsic_matrices, scale_factor=1.0):
    """
    Convert VGGT camera-from-world extrinsics to a metric camera trajectory.

    The evaluation pipeline appends the benchmark first frame after the predicted
    video frames for scale alignment, so the last camera is excluded from the
    returned predicted trajectory.
    """
    if torch is not None and isinstance(extrinsic_matrices, torch.Tensor):
        extrinsic_matrices = extrinsic_matrices.detach().cpu().numpy()

    extrinsic_matrices = np.asarray(extrinsic_matrices, dtype=np.float32)
    if extrinsic_matrices.ndim == 4:
        extrinsic_matrices = extrinsic_matrices[0]
    if extrinsic_matrices.ndim != 3 or extrinsic_matrices.shape[1:] != (3, 4):
        raise ValueError(f"Expected extrinsics with shape (N, 3, 4), got {extrinsic_matrices.shape}")
    if extrinsic_matrices.shape[0] < 2:
        raise ValueError("At least two cameras are required to build a predicted trajectory")

    r_w2c = extrinsic_matrices[:, :3, :3]
    t_w2c = extrinsic_matrices[:, :3, 3]
    r_c2w = np.transpose(r_w2c, (0, 2, 1))
    t_c2w = -np.matmul(r_c2w, t_w2c[:, :, None])[:, :, 0]

    trajectory_raw = t_c2w * scale_factor
    trajectory = np.zeros((extrinsic_matrices.shape[0] - 1, 3), dtype=np.float32)
    trajectory[:, 0] = trajectory_raw[:-1, 0]
    trajectory[:, 1] = trajectory_raw[:-1, 2]
    trajectory[:, 2] = trajectory_raw[:-1, 1]

    orientations = r_c2w
    trajectory_2d = trajectory[:, :2]
    return trajectory, orientations, trajectory_2d


def load_ground_truth_trajectory(segment_path, segment_name):
    """Load the full benchmark trajectory from a segment CSV."""
    gt_csv_path = os.path.join(segment_path, f"{segment_name}_trajectory.csv")
    if not os.path.exists(gt_csv_path):
        print(f"Warning: Ground truth trajectory file not found: {gt_csv_path}")
        return None

    try:
        gt_df = pd.read_csv(gt_csv_path)
        if "x" not in gt_df.columns or "y" not in gt_df.columns:
            print(f"Warning: GT trajectory CSV missing x/y columns: {gt_csv_path}")
            return None

        gt_trajectory_2d = gt_df[["x", "y"]].values
        gt_trajectory_2d = gt_trajectory_2d[~np.isnan(gt_trajectory_2d).any(axis=1)]
        print(f"Loaded GT trajectory with {len(gt_trajectory_2d)} points from {os.path.basename(gt_csv_path)}")
        return gt_trajectory_2d

    except Exception as exc:
        print(f"Error loading GT trajectory from {gt_csv_path}: {exc}")
        return None
