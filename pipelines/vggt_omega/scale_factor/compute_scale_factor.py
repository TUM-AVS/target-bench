"""
VGGT-Omega adapter for target-bench trajectory evaluation.

The official VGGT-Omega package is loaded lazily so static checks and --help
do not require the model repository or checkpoint to be installed.
"""

import os
import sys

import cv2
import numpy as np
import torch
import yaml


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../../"))


def load_config(config_path=None):
    """Load VGGT-Omega pipeline configuration."""
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "pipelines/vggt_omega/configs/vggt_omega_pcd_configs.yml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loaded VGGT-Omega configuration from {config_path}")
    return config


def _candidate_source_paths(config):
    paths = []
    model_cfg = (config or {}).get("model", {})

    paths.extend(model_cfg.get("model_source_paths", []) or [])

    src_dir = os.environ.get("VGGT_OMEGA_SRC_DIR")
    if src_dir:
        paths.append(src_dir)

    repo_dir = os.environ.get("VGGT_OMEGA_REPO_DIR")
    if repo_dir:
        paths.append(repo_dir)

    paths.extend([
        os.path.join(PROJECT_ROOT, "models/vggt-omega"),
        os.path.join(PROJECT_ROOT, "models/vggt_omega"),
    ])

    return paths


def _add_vggt_omega_source_paths(config):
    """Allow using a local official checkout without installing it."""
    for path in _candidate_source_paths(config):
        if path and os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)


def _resolve_project_path(path):
    expanded = os.path.expandvars(os.path.expanduser(path))
    if not os.path.isabs(expanded):
        expanded = os.path.join(PROJECT_ROOT, expanded)
    return os.path.abspath(expanded)


def _checkpoint_dir(config=None):
    model_cfg = (config or {}).get("model", {})
    env_root = os.environ.get("TARGET_BENCH_CHECKPOINT_DIR")
    if env_root:
        return os.path.join(_resolve_project_path(env_root), "vggt_omega")

    if model_cfg.get("checkpoint_dir"):
        return _resolve_project_path(model_cfg["checkpoint_dir"])

    root = model_cfg.get("checkpoint_root") or "checkpoints"
    return os.path.join(_resolve_project_path(root), "vggt_omega")


def _candidate_checkpoint_paths(config, checkpoint_path=None):
    paths = []
    if checkpoint_path:
        paths.append(checkpoint_path)

    env_checkpoint = os.environ.get("VGGT_OMEGA_CHECKPOINT")
    if env_checkpoint:
        paths.append(env_checkpoint)

    model_cfg = (config or {}).get("model", {})
    config_checkpoint = model_cfg.get("checkpoint_path")
    if config_checkpoint:
        paths.append(config_checkpoint)

    paths.extend([
        os.path.join(_checkpoint_dir(config), model_cfg.get("checkpoint_filename", "vggt_omega_1b_512.pt")),
        os.path.join(PROJECT_ROOT, "models/vggt_omega/VGGT-Omega-1B-512/model.pt"),
        os.path.join(PROJECT_ROOT, "models/vggt-omega/checkpoints/VGGT-Omega-1B-512/model.pt"),
    ])

    return paths


def resolve_checkpoint_path(config, checkpoint_path=None):
    """Resolve a local VGGT-Omega checkpoint or download the official HF weight."""
    for path in _candidate_checkpoint_paths(config, checkpoint_path):
        expanded = _resolve_project_path(path)
        if os.path.isfile(expanded):
            return expanded

    model_cfg = (config or {}).get("model", {})
    repo_id = model_cfg.get("hf_repo_id", "facebook/vggt-omega")
    filename = model_cfg.get("checkpoint_filename", "vggt_omega_1b_512.pt")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "VGGT-Omega checkpoint not found locally and huggingface_hub is not installed. "
            "Set VGGT_OMEGA_CHECKPOINT or model.checkpoint_path to a local model.pt."
        ) from exc

    try:
        local_dir = _checkpoint_dir(config)
        os.makedirs(local_dir, exist_ok=True)
        return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    except Exception as exc:
        searched = "\n  - ".join(_candidate_checkpoint_paths(config, checkpoint_path))
        raise RuntimeError(
            "VGGT-Omega checkpoint not found locally and automatic Hugging Face "
            f"download failed for {repo_id}/{filename}. Make sure your HF account "
            "has access and run `hf auth login`, or set VGGT_OMEGA_CHECKPOINT/"
            "model.checkpoint_path to a local checkpoint.\n"
            f"Searched:\n  - {searched}"
        ) from exc


def setup_model(config=None, checkpoint_path=None, device=None):
    """
    Initialize VGGT-Omega from the official package and checkpoint.

    Args:
        config: VGGT-Omega pipeline config dict.
        checkpoint_path: Optional local checkpoint override.
        device: Optional torch device string.

    Returns:
        (model, device)
    """
    if config is None:
        config = load_config()

    if not torch.cuda.is_available() and device is None:
        raise RuntimeError(
            "VGGT-Omega inference requires CUDA in the official implementation. "
            "Run on a CUDA machine or adapt the official package for CPU."
        )

    _add_vggt_omega_source_paths(config)

    try:
        from vggt_omega.models import VGGTOmega
    except ImportError as exc:
        raise RuntimeError(
            "VGGT-Omega is not importable. Install the official repo with "
            "`pip install -e /path/to/vggt-omega` or set VGGT_OMEGA_REPO_DIR/"
            "VGGT_OMEGA_SRC_DIR to a local checkout."
        ) from exc

    model_cfg = config.get("model", {})
    ckpt_path = resolve_checkpoint_path(config, checkpoint_path=checkpoint_path)
    device = device or "cuda"

    print(f"Loading VGGT-Omega checkpoint: {ckpt_path}")
    model = VGGTOmega(enable_alignment=bool(model_cfg.get("enable_alignment", False))).eval()
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, device


def extract_frames(video_path, output_dir, num_frames=9):
    """
    Extract uniformly sampled frames from a video.

    The sampling behavior matches the other target-bench evaluators: indices
    are distributed over the full video duration, including the first and last
    readable frames.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    sample_count = min(int(num_frames), total_frames)
    frame_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

    frame_paths = []
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    output_frames_dir = os.path.join(output_dir, video_stem)
    os.makedirs(output_frames_dir, exist_ok=True)

    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            continue

        frame_path = os.path.join(output_frames_dir, f"frame_{idx:02d}_idx{int(frame_idx):04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)

    cap.release()

    if not frame_paths:
        raise RuntimeError(f"Could not extract any frames from: {video_path}")

    return frame_paths


def run_vggt_omega_inference(model, image_paths, config, device=None):
    """Run VGGT-Omega camera inference and return extrinsics/intrinsics."""
    _add_vggt_omega_source_paths(config)

    try:
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera
    except ImportError as exc:
        raise RuntimeError(
            "VGGT-Omega utilities are not importable. Install the official repo "
            "or set VGGT_OMEGA_REPO_DIR/VGGT_OMEGA_SRC_DIR."
        ) from exc

    preprocess_cfg = config.get("preprocess", {})
    device = device or next(model.parameters()).device

    images = load_and_preprocess_images(
        image_paths,
        mode=preprocess_cfg.get("mode", "balanced"),
        image_resolution=int(preprocess_cfg.get("image_resolution", 512)),
        patch_size=int(preprocess_cfg.get("patch_size", 16)),
    ).to(device)

    with torch.inference_mode():
        predictions = model(images)
        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
        )

    return extrinsics, intrinsics
