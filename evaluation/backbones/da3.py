"""Depth Anything 3 backbone adapter."""

import os

import numpy as np

from .utils import load_module_from_path


class DA3Backbone:
    name = "da3"
    display_name = "DA3"

    def __init__(self, project_root, config_path=None, model_id=None, checkpoint_path=None):
        self.project_root = project_root
        self.config_path = config_path or os.path.join(
            project_root, "pipelines/da3/configs/da3_pcd_configs.yml"
        )
        self.model_id = model_id
        self.output_subdir = "da3"
        self.eval_config_path = os.path.join(project_root, "evaluation/config/da3_config.yaml")
        self.config = None
        self.num_video_frames = 9
        self.model = None
        self.device = None
        self.scale = None
        self.extrinsic = None

    def setup(self):
        self.scale = load_module_from_path(
            "target_bench_da3_scale_factor",
            os.path.join(self.project_root, "pipelines/da3/scale_factor/compute_scale_factor.py"),
        )
        self.extrinsic = load_module_from_path(
            "target_bench_da3_extrinsic_path",
            os.path.join(self.project_root, "pipelines/da3/extrinsic_path/extract_extrinsic_path.py"),
        )

        self.config = self.scale.load_config(self.config_path)
        self.num_video_frames = int(self.config.get("video", {}).get("num_frames", 9))
        self.model, self.device = self.scale.setup_model(self.config, model_id=self.model_id)
        active_model_id = self.model_id or self.config.get("model", {}).get("model_id", "DA3-GIANT-1.1")
        print(f"Model loaded successfully on {self.device}: {active_model_id}")

    def run(self, segment_dir, segment_name, video_path, metadata, output_dir):
        first_frame_path = os.path.join(segment_dir, f"{segment_name}_first_frame.png")
        real_displacement = float(metadata["distance_first_to_current"]["distance_meters"])

        temp_frame_dir = os.path.join(output_dir, "video_frames")
        os.makedirs(temp_frame_dir, exist_ok=True)
        predicted_image_paths = self.scale.extract_frames(
            video_path,
            temp_frame_dir,
            num_frames=self.num_video_frames,
        )

        camera_paths = list(predicted_image_paths)
        camera_paths.append(first_frame_path)

        extrinsic = self.scale.run_da3_inference(self.model, camera_paths, self.config, self.device)
        camera_centers = self.scale.camera_centers_from_extrinsics(extrinsic)
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
