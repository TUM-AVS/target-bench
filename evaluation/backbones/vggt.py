"""VGGT backbone adapter."""

import contextlib
import os

import numpy as np
import torch

from .utils import add_path, load_module_from_path


class VGGTBackbone:
    name = "vggt"
    display_name = "VGGT"

    def __init__(self, project_root, config_path=None, model_id=None, checkpoint_path=None):
        self.project_root = project_root
        self.config_path = config_path or os.path.join(
            project_root, "pipelines/vggt/configs/vggt_pcd_configs.yml"
        )
        self.checkpoint_path = checkpoint_path
        self.output_subdir = "vggt"
        self.eval_config_path = os.path.join(project_root, "evaluation/config/vggt_config.yaml")
        self.model = None
        self.device = None
        self.dtype = None
        self.scale = None
        self.extrinsic = None
        self.load_and_preprocess_images = None
        self.pose_encoding_to_extri_intri = None

    def setup(self):
        add_path(os.path.join(self.project_root, "models/vggt"))

        self.scale = load_module_from_path(
            "target_bench_vggt_scale_factor",
            os.path.join(self.project_root, "pipelines/vggt/scale_factor/compute_scale_factor.py"),
        )
        self.extrinsic = load_module_from_path(
            "target_bench_vggt_extrinsic_path",
            os.path.join(self.project_root, "pipelines/vggt/extrinsic_path/extract_extrinsic_path.py"),
        )

        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        self.load_and_preprocess_images = load_and_preprocess_images
        self.pose_encoding_to_extri_intri = pose_encoding_to_extri_intri

        config = self.scale.load_config(self.config_path)
        self.model, self.device, self.dtype = self.scale.setup_model(
            config=config,
            checkpoint_path=self.checkpoint_path,
        )
        print(f"Model loaded successfully on {self.device} with dtype {self.dtype}")

    def run(self, segment_dir, segment_name, video_path, metadata, output_dir):
        first_frame_path = os.path.join(segment_dir, f"{segment_name}_first_frame.png")
        real_displacement = float(metadata["distance_first_to_current"]["distance_meters"])

        temp_frame_dir = os.path.join(output_dir, "video_frames")
        os.makedirs(temp_frame_dir, exist_ok=True)
        predicted_image_paths = self.scale.extract_frames(video_path, temp_frame_dir)

        camera_paths = list(predicted_image_paths)
        camera_paths.append(first_frame_path)

        images = self.load_and_preprocess_images(camera_paths).to(self.device)

        autocast_context = (
            torch.cuda.amp.autocast(dtype=self.dtype)
            if self.device == "cuda"
            else contextlib.nullcontext()
        )

        with torch.no_grad():
            with autocast_context:
                self.model(images)
                images = images[None]
                aggregated_tokens_list, _ps_idx = self.model.aggregator(images)

            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, _intrinsic = self.pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            camera1_position = extrinsic[0, 0, :3, 3]
            camera2_position = extrinsic[0, -1, :3, 3]
            camera_displacement = torch.norm(camera1_position - camera2_position).item()

        if camera_displacement == 0:
            raise RuntimeError("Zero camera displacement")

        scale_factor = real_displacement / camera_displacement
        trajectory, _orientations, trajectory_2d = self.extrinsic.extract_trajectory_from_extrinsics(
            extrinsic.cpu().numpy(),
            scale_factor,
        )
        trajectory_length = float(np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)))

        return {
            "scale_factor": float(scale_factor),
            "real_displacement": real_displacement,
            "camera_displacement": float(camera_displacement),
            "trajectory_length": trajectory_length,
            "trajectory": trajectory,
            "trajectory_2d": trajectory_2d,
        }
