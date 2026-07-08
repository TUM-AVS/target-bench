"""
Depth Anything 3 adapter for target-bench trajectory evaluation.

This module mirrors the small API surface used by the VGGT evaluation path:
`load_config`, `setup_model`, and `extract_frames`. It also exposes helpers for
running DA3 pose inference and converting OpenCV/Colmap w2c matrices to camera
centers.
"""

import os
import sys
import types

import cv2
import numpy as np
import torch
import yaml


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../../"))


MODEL_ID_ALIASES = {
    "DA3-GIANT-1.1": "depth-anything/DA3-GIANT-1.1",
    "DA3NESTED-GIANT-LARGE-1.1": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    "DA3-GIANT": "depth-anything/DA3-GIANT",
    "DA3NESTED-GIANT-LARGE": "depth-anything/DA3NESTED-GIANT-LARGE",
}


def load_config(config_path=None):
    """Load DA3 pipeline configuration."""
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "pipelines/da3/configs/da3_pcd_configs.yml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loaded DA3 configuration from {config_path}")
    return config


def resolve_model_id(model_id):
    """Resolve shorthand DA3 model names to Hugging Face repo IDs."""
    return MODEL_ID_ALIASES.get(model_id, model_id)


def _resolve_project_path(path):
    expanded = os.path.expandvars(os.path.expanduser(path))
    if not os.path.isabs(expanded):
        expanded = os.path.join(PROJECT_ROOT, expanded)
    return os.path.abspath(expanded)


def _checkpoint_dir(config=None):
    model_cfg = (config or {}).get("model", {})
    env_root = os.environ.get("TARGET_BENCH_CHECKPOINT_DIR")
    if env_root:
        return os.path.join(_resolve_project_path(env_root), "da3")

    if model_cfg.get("checkpoint_dir"):
        return _resolve_project_path(model_cfg["checkpoint_dir"])

    root = model_cfg.get("checkpoint_root") or "checkpoints"
    return os.path.join(_resolve_project_path(root), "da3")


def _candidate_source_paths(config):
    paths = []
    model_cfg = (config or {}).get("model", {})

    paths.extend(model_cfg.get("model_source_paths", []) or [])

    da3_src_dir = os.environ.get("DA3_SRC_DIR")
    if da3_src_dir:
        paths.append(da3_src_dir)

    da3_repo_dir = os.environ.get("DA3_REPO_DIR")
    if da3_repo_dir:
        paths.append(os.path.join(da3_repo_dir, "src"))

    paths.extend([
        os.path.join(PROJECT_ROOT, "models/da3/src"),
        os.path.join(PROJECT_ROOT, "models/da3"),
        os.path.join(PROJECT_ROOT, "models/Depth-Anything-3/src"),
        os.path.join(PROJECT_ROOT, "models/Depth-Anything-3"),
        os.path.join(PROJECT_ROOT, "../Depth-Anything-3/src"),
        os.path.join(PROJECT_ROOT, "../Depth-Anything-3"),
    ])

    return paths


def _add_da3_source_paths(config):
    """Allow using a local DA3 checkout without installing it as a package."""
    for path in _candidate_source_paths(config):
        if path and os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)


def _as_homogeneous_extrinsics(extrinsics):
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"Unsupported extrinsic shape: {extrinsics.shape}")

    padded = np.zeros((*extrinsics.shape[:-2], 4, 4), dtype=extrinsics.dtype)
    padded[..., :3, :4] = extrinsics
    padded[..., 3, 3] = 1.0
    return padded


def _umeyama_similarity(src, dst):
    """Estimate Sim(3) that maps src points to dst points."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected matching Nx3 points, got {src.shape} and {dst.shape}")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    cov = dst_centered.T @ src_centered / max(len(src), 1)
    u, singular_values, vt = np.linalg.svd(cov)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1

    rotation = u @ correction @ vt
    variance = np.sum(src_centered * src_centered) / max(len(src), 1)
    scale = float(np.trace(np.diag(singular_values) @ correction) / max(variance, 1e-12))
    translation = dst_mean - scale * (rotation @ src_mean)
    return rotation, translation, scale


def _install_da3_eval_import_stubs():
    """
    Avoid pulling DA3 demo/export/benchmark dependencies into the eval runtime.

    The official DA3 API imports export and pose-alignment helpers at module
    import time. Target-Bench DA3 eval only needs inference outputs and does not
    request exports or pass reference extrinsics, so these lightweight stubs keep
    requirements scoped to inference.
    """
    if "depth_anything_3.utils.export" not in sys.modules:
        export_module = types.ModuleType("depth_anything_3.utils.export")

        def export(*args, **kwargs):
            raise RuntimeError("DA3 export is disabled in the Target-Bench eval runtime.")

        export_module.export = export
        sys.modules["depth_anything_3.utils.export"] = export_module

    if "depth_anything_3.utils.parallel_utils" not in sys.modules:
        parallel_module = types.ModuleType("depth_anything_3.utils.parallel_utils")

        def async_call(fn):
            return fn

        def parallel_execution(*args, action, **kwargs):
            def get_length():
                for arg in args:
                    if isinstance(arg, list):
                        return len(arg)
                for value in kwargs.values():
                    if isinstance(value, list):
                        return len(value)
                raise ValueError("parallel_execution requires at least one list input.")

            length = get_length()
            results = []
            control_keys = {"num_processes", "print_progress", "sequential", "async_return", "desc"}
            action_kwargs = {key: value for key, value in kwargs.items() if key not in control_keys}

            for idx in range(length):
                item_args = [
                    arg[idx] if isinstance(arg, list) and len(arg) == length else arg
                    for arg in args
                ]
                item_kwargs = {
                    key: value[idx] if isinstance(value, list) and len(value) == length else value
                    for key, value in action_kwargs.items()
                }
                results.append(action(*item_args, **item_kwargs))
            return results

        parallel_module.async_call = async_call
        parallel_module.parallel_execution = parallel_execution
        sys.modules["depth_anything_3.utils.parallel_utils"] = parallel_module

    if "depth_anything_3.utils.pose_align" not in sys.modules:
        pose_align_module = types.ModuleType("depth_anything_3.utils.pose_align")

        def align_poses_umeyama(
            ext_ref,
            ext_est,
            return_aligned=False,
            ransac=False,
            sub_n=None,
            inlier_thresh=None,
            ransac_max_iters=10,
            random_state=None,
        ):
            ref_pose = np.linalg.inv(_as_homogeneous_extrinsics(ext_ref))
            est_pose = np.linalg.inv(_as_homogeneous_extrinsics(ext_est))
            rotation, translation, scale = _umeyama_similarity(
                est_pose[:, :3, 3],
                ref_pose[:, :3, 3],
            )

            if not return_aligned:
                return rotation, translation, scale

            aligned_pose = est_pose.copy()
            aligned_pose[:, :3, :3] = rotation @ est_pose[:, :3, :3]
            aligned_pose[:, :3, 3] = scale * (rotation @ est_pose[:, :3, 3].T).T + translation
            aligned_ext = np.linalg.inv(aligned_pose)
            return rotation, translation, scale, aligned_ext

        def batch_align_poses_umeyama(ext_ref, ext_est):
            rotations, translations, scales = [], [], []
            for batch_idx in range(ext_ref.shape[0]):
                rotation, translation, scale = align_poses_umeyama(
                    ext_ref[batch_idx].detach().cpu().numpy(),
                    ext_est[batch_idx].detach().cpu().numpy(),
                )
                rotations.append(torch.as_tensor(rotation, dtype=ext_ref.dtype, device=ext_ref.device))
                translations.append(torch.as_tensor(translation, dtype=ext_ref.dtype, device=ext_ref.device))
                scales.append(torch.as_tensor(scale, dtype=ext_ref.dtype, device=ext_ref.device))
            return torch.stack(rotations), torch.stack(translations), torch.stack(scales)

        pose_align_module.align_poses_umeyama = align_poses_umeyama
        pose_align_module.batch_align_poses_umeyama = batch_align_poses_umeyama
        sys.modules["depth_anything_3.utils.pose_align"] = pose_align_module


def setup_model(config=None, model_id=None, device=None):
    """
    Initialize a Depth Anything 3 model.

    Args:
        config: DA3 pipeline config dict. If omitted, the default config is used.
        model_id: Optional override for `config['model']['model_id']`.
        device: Optional torch device string.

    Returns:
        (model, device)
    """
    if config is None:
        config = load_config()

    _add_da3_source_paths(config)
    _install_da3_eval_import_stubs()

    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        raise RuntimeError(
            "Depth Anything 3 is not importable. Install the official repo with "
            "`git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git models/da3`, "
            "then run `bash set_env.sh install`, or set DA3_REPO_DIR/DA3_SRC_DIR."
        ) from exc

    model_cfg = config.get("model", {})
    hf_model_id = resolve_model_id(model_id or model_cfg.get("model_id", "DA3-GIANT-1.1"))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = _checkpoint_dir(config)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Loading DA3 model: {hf_model_id}")
    print(f"Using DA3 checkpoint cache: {cache_dir}")
    model = DepthAnything3.from_pretrained(hf_model_id, cache_dir=cache_dir)
    model = model.to(device)
    model.eval()

    return model, device


def extract_frames(video_path, output_dir, num_frames=9):
    """
    Extract uniformly sampled frames from a video.

    The sampling behavior intentionally matches the VGGT path: `num_frames`
    indices are distributed over the full video duration, including the first
    and last available frames.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    sample_count = min(num_frames, total_frames)
    frame_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

    frame_paths = []
    video_stem = os.path.splitext(os.path.basename(video_path))[0]

    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            continue

        frame_path = os.path.join(output_dir, f"{video_stem}_frame_{idx:03d}.png")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)

    cap.release()

    if not frame_paths:
        raise RuntimeError(f"Could not extract any frames from: {video_path}")

    return frame_paths


def _camera_to_world_to_world_to_camera(extrinsics):
    """Convert camera-to-world pose matrices to world-to-camera extrinsics."""
    r_c2w = extrinsics[:, :3, :3]
    t_c2w = extrinsics[:, :3, 3]
    r_w2c = np.transpose(r_c2w, (0, 2, 1))
    t_w2c = -np.matmul(r_w2c, t_c2w[:, :, None])[:, :, 0]

    converted = extrinsics.copy()
    converted[:, :3, :3] = r_w2c
    converted[:, :3, 3] = t_w2c
    return converted


def run_da3_inference(model, image_paths, config, device=None):
    """Run DA3 inference and return extrinsic matrices as a numpy array."""
    infer_cfg = config.get("inference", {})
    use_ray_pose = bool(infer_cfg.get("use_ray_pose", False))

    with torch.no_grad():
        prediction = model.inference(
            image_paths,
            process_res=int(infer_cfg.get("process_res", 504)),
            process_res_method=infer_cfg.get("process_res_method", "upper_bound_resize"),
            use_ray_pose=use_ray_pose,
            ref_view_strategy=infer_cfg.get("ref_view_strategy", "first"),
            align_to_input_ext_scale=bool(infer_cfg.get("align_to_input_ext_scale", True)),
        )

    extrinsics = getattr(prediction, "extrinsics", None)
    if extrinsics is None:
        raise RuntimeError("DA3 prediction did not contain extrinsics.")

    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    if extrinsics.ndim == 4:
        extrinsics = extrinsics[0]

    if extrinsics.ndim != 3 or extrinsics.shape[1] < 3 or extrinsics.shape[2] < 4:
        raise RuntimeError(f"Unexpected DA3 extrinsics shape: {extrinsics.shape}")

    # DA3's ray-pose path stores camera-to-world poses in `prediction.extrinsics`;
    # the default camera decoder stores world-to-camera extrinsics.
    if use_ray_pose:
        extrinsics = _camera_to_world_to_world_to_camera(extrinsics)

    return extrinsics


def camera_centers_from_extrinsics(extrinsic_matrices):
    """
    Convert OpenCV/Colmap world-to-camera extrinsics to camera centers.

    Args:
        extrinsic_matrices: array with shape (N, 3/4, 4) or (B, N, 3/4, 4)

    Returns:
        np.ndarray with shape (N, 3)
    """
    extrinsic_matrices = np.asarray(extrinsic_matrices, dtype=np.float32)

    if extrinsic_matrices.ndim == 4:
        extrinsic_matrices = extrinsic_matrices[0]

    if extrinsic_matrices.ndim != 3 or extrinsic_matrices.shape[1] < 3 or extrinsic_matrices.shape[2] < 4:
        raise ValueError(f"Unexpected extrinsics shape: {extrinsic_matrices.shape}")

    r_w2c = extrinsic_matrices[:, :3, :3]
    t_w2c = extrinsic_matrices[:, :3, 3]
    r_c2w = np.transpose(r_w2c, (0, 2, 1))
    centers = -np.matmul(r_c2w, t_w2c[:, :, None])[:, :, 0]

    return centers
