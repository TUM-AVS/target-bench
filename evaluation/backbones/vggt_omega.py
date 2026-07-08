"""VGGT-Omega backbone adapter."""

import os

import numpy as np

from .utils import load_module_from_path


def _camera_centers_from_extrinsics(extrinsic_matrices):
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(extrinsic_matrices, torch.Tensor):
        extrinsic_matrices = extrinsic_matrices.detach().cpu().numpy()

    extrinsic_matrices = np.asarray(extrinsic_matrices, dtype=np.float32)
    if extrinsic_matrices.ndim == 4:
        extrinsic_matrices = extrinsic_matrices[0]

    r_w2c = extrinsic_matrices[:, :3, :3]
    t_w2c = extrinsic_matrices[:, :3, 3]
    r_c2w = np.transpose(r_w2c, (0, 2, 1))
    return -np.matmul(r_c2w, t_w2c[:, :, None])[:, :, 0]


class VGGTOmegaBackbone:
    name = "vggt_omega"
    display_name = "VGGT-Omega"

    def __init__(self, project_root, config_path=None, model_id=None, checkpoint_path=None):
        self.project_root = project_root
        self.config_path = config_path or os.path.join(
            project_root, "pipelines/vggt_omega/configs/vggt_omega_pcd_configs.yml"
        )
        self.checkpoint_path = checkpoint_path
        self.output_subdir = "vggt_omega"
        self.eval_config_path = os.path.join(project_root, "evaluation/config/vggt_omega_config.yaml")
        self.config = None
        self.num_video_frames = 9
        self.model = None
        self.device = None
        self.scale = None
        self.extrinsic = None

    def setup(self):
        self.scale = load_module_from_path(
            "target_bench_vggt_omega_scale_factor",
            os.path.join(self.project_root, "pipelines/vggt_omega/scale_factor/compute_scale_factor.py"),
        )
        self.extrinsic = load_module_from_path(
            "target_bench_vggt_omega_extrinsic_path",
            os.path.join(self.project_root, "pipelines/vggt_omega/extrinsic_path/extract_extrinsic_path.py"),
        )

        self.config = self.scale.load_config(self.config_path)
        self.num_video_frames = int(self.config.get("video", {}).get("num_frames", 9))
        self.model, self.device = self.scale.setup_model(self.config, checkpoint_path=self.checkpoint_path)
        model_name = self.config.get("model", {}).get("model_name", "VGGT-Omega")
        print(f"Model loaded successfully on {self.device}: {model_name}")

    def run(self, segment_dir, segment_name, video_path, metadata, output_dir):
        first_frame_path = os.path.join(segment_dir, f"{segment_name}_first_frame.png")
        real_displacement = float(metadata["distance_first_to_current"]["distance_meters"])

        temp_frame_dir = os.path.join(output_dir, "video_frames")
        predicted_image_paths = self.scale.extract_frames(
            video_path,
            temp_frame_dir,
            num_frames=self.num_video_frames,
        )

        camera_paths = list(predicted_image_paths)
        camera_paths.append(first_frame_path)

        extrinsic, _intrinsic = self.scale.run_vggt_omega_inference(
            self.model,
            camera_paths,
            self.config,
            self.device,
        )
        camera_centers = _camera_centers_from_extrinsics(extrinsic)
        camera_displacement = float(np.linalg.norm(camera_centers[0] - camera_centers[-1]))
        if camera_displacement == 0:
            raise RuntimeError("Zero camera displacement")

        scale_factor = real_displacement / camera_displacement
        trajectory, _orientations, trajectory_2d = self.extrinsic.extract_trajectory_from_extrinsics(
            extrinsic,
            scale_factor,
        )
        trajectory_length = float(np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)))

        return {
            "scale_factor": float(scale_factor),
            "real_displacement": real_displacement,
            "camera_displacement": camera_displacement,
            "trajectory_length": trajectory_length,
            "trajectory": trajectory,
            "trajectory_2d": trajectory_2d,
        }
