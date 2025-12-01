"""
Scale factor computation using SpaTrackerV2 for EgoWalk_samples dataset.

This module adapts the compute_scale_factor.py functionality to use SpaTrackerV2 
instead of VGGT for pose estimation and depth prediction.

Key differences from VGGT version:
1. Uses SpaTrackerV2/VGGT4Track models for pose and depth estimation
2. Integrates tracking capabilities for better temporal consistency
3. Maintains compatibility with EgoWalk dataset structure

Key functionality:
1. Processes EgoWalk segments using VGGT4Track models
2. Extracts camera trajectories from pose predictions with metric scale factors  
3. Handles video frame extraction and processing
4. Computes scale factors by comparing predicted vs real displacements
"""

import torch
import numpy as np
import os
import pickle
import glob
import cv2
import json
import yaml
import struct
from pathlib import Path
import pandas as pd
from PIL import Image
import sys
# Define project root
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../../"))
sys.path.append(os.path.join(PROJECT_ROOT, 'models/spatracker'))

from tqdm.auto import tqdm
import requests
import copy
import decord
import torchvision.transforms as T

# SpaTrackerV2 imports
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from models.SpaTrackV2.models.vggt4track.utils.pose_enc import pose_encoding_to_extri_intri
from models.SpaTrackV2.models.predictor import Predictor
from models.SpaTrackV2.models.utils import get_points_on_a_grid

# Sky segmentation imports
try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

def load_config(config_path="spatracker_pcd_configs.yml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using default settings.")
        # Return default config
        return {
            'model': {'confidence_threshold': 0.5, 'apply_sky_mask': True, 'track_mode': 'offline'},
            'post_processing': {
                'fix_coordinates': {'enabled': True},
                'filter_points': {'enabled': True, 'min_z': -2.0, 'max_z': 0.5},
                'scale_coordinates': {'enabled': True, 'scale_factor': 1.1}
            },
            'paths': {  
                'segments_dir': os.path.join(PROJECT_ROOT, "dataset/EgoWalk_samples"),
                'output_base_path': os.path.join(PROJECT_ROOT, "pipelines/spatracker/output"),
                'camera_params_json': os.path.join(PROJECT_ROOT, "camera_para.json")
            },
            'cameras': {'camera_order': ['CAM_F0', 'CAM_L0', 'CAM_R0']},
            'spatracker': {
                'model_name_offline': "Yuxihenry/SpatialTrackerV2-Offline",
                'model_name_online': "Yuxihenry/SpatialTrackerV2-Online",
                'vggt_model_name': "Yuxihenry/SpatialTrackerV2_Front",
                'fps': 1,
                'max_frames_offline': 15,
                'max_frames_online': 300,
                'min_video_scale': 224,
                'grid_size': 10,
                'vo_points': 756,
                'track_mode': 'offline'
            }
        }

def setup_spatracker_models(config):
    """Initialize SpaTrackerV2 and VGGT4Track models."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    spatracker_config = config.get('spatracker', {})
    track_mode = config.get('model', {}).get('track_mode', 'offline')
    
    print("Loading SpaTrackerV2 and VGGT4Track models...")
    
    # Load VGGT4Track model for pose and depth estimation
    vggt4track_model = VGGT4Track.from_pretrained(spatracker_config.get('vggt_model_name', "Yuxihenry/SpatialTrackerV2_Front"))
    vggt4track_model.eval()
    vggt4track_model = vggt4track_model.to(device)
    
    # Load SpaTracker model based on mode
    if track_mode == "offline":
        spatracker_model = Predictor.from_pretrained(spatracker_config.get('model_name_offline', "Yuxihenry/SpatialTrackerV2-Offline"))
    else:
        spatracker_model = Predictor.from_pretrained(spatracker_config.get('model_name_online', "Yuxihenry/SpatialTrackerV2-Online"))
    
    spatracker_model.eval()
    spatracker_model = spatracker_model.to(device)
    
    return vggt4track_model, spatracker_model, device, dtype

def extract_frames(video_path, output_dir):
    """Extract frames from video file."""
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return []

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    print(f"FPS: {fps}, Duration: {duration:.2f}s")

    # Ensure output directory exists
    output_frames_dir = os.path.join(output_dir, os.path.basename(video_path).split('.')[0])
    os.makedirs(output_frames_dir, exist_ok=True)

    frames_list = []

    # Extract frames at 0s, 1s, ..., 8s
    for sec in range(9):  # 0 to 8 inclusive
        frame_idx = min(int(sec * fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            filename = os.path.join(output_frames_dir, f"frame_{sec:02d}s.jpg")
            frames_list.append(filename)
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
        else:
            print(f"Warning: Could not read frame at {sec}s")

    cap.release()
    print("Done.")
    return frames_list

def process_video_with_spatracker(video_path, vggt4track_model, spatracker_model, config, device, dtype):
    """
    Process video using SpaTrackerV2 models to get poses and depth.
    
    Args:
        video_path: Path to video file
        vggt4track_model: VGGT4Track model for pose/depth estimation
        spatracker_model: SpaTracker model for tracking
        config: Configuration dictionary
        device: Computing device
        dtype: Data type for computation
        
    Returns:
        dict: Dictionary containing extrinsics, intrinsics, depths, and other predictions
    """
    spatracker_config = config.get('spatracker', {})
    fps = spatracker_config.get('fps', 1)
    track_mode = config.get('model', {}).get('track_mode', 'offline')
    max_frames = spatracker_config.get('max_frames_offline', 80) if track_mode == 'offline' else spatracker_config.get('max_frames_online', 300)
    min_scale = spatracker_config.get('min_video_scale', 224)
    
    # Load video using decord
    video_reader = decord.VideoReader(video_path)
    video_tensor = torch.from_numpy(video_reader.get_batch(range(len(video_reader))).asnumpy()).permute(0, 3, 1, 2)
    
    # Resize to ensure minimum side meets requirements
    h, w = video_tensor.shape[2:]
    scale = max(min_scale / h, min_scale / w)
    if scale > 1:
        new_h, new_w = int(h * scale), int(w * scale)
        video_tensor = T.Resize((new_h, new_w))(video_tensor)
    
    # Sample frames and limit by max_frames
    video_tensor = video_tensor[::fps].float()[:max_frames]
    video_tensor = video_tensor.to(device)
    
    print(f"Processing video tensor shape: {video_tensor.shape}")
    
    # Process with VGGT4Track model
    video_tensor_processed = preprocess_image(video_tensor)[None]
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict poses, depths, and point maps
            predictions = vggt4track_model(video_tensor_processed / 255)
            extrinsic = predictions["poses_pred"]  # Camera poses
            intrinsic = predictions["intrs"]       # Camera intrinsics  
            depth_map = predictions["points_map"][..., 2]  # Depth maps
            depth_conf = predictions["unc_metric"]         # Depth confidence
    
    # Convert to numpy for processing
    extrinsic_np = extrinsic.squeeze().cpu().numpy()
    intrinsic_np = intrinsic.squeeze().cpu().numpy()
    depth_tensor_np = depth_map.squeeze().cpu().numpy()
    unc_metric_np = depth_conf.squeeze().cpu().numpy() > 0.5
    
    # Prepare results dictionary
    results = {
        'extrinsics': extrinsic_np,
        'intrinsics': intrinsic_np,
        'depths': depth_tensor_np,
        'unc_metric': unc_metric_np,
        'video': video_tensor_processed.squeeze().cpu().numpy() / 255,
        'predictions': predictions
    }
    
    return results

def sample_video_frames(video: torch.Tensor, current_frame: int, num_samples: int):
    """
    Uniformly sample `num_samples` frames from [0, T-1],
    always including first (0), current (by proportion), and last (T-1).
    
    Returns:
        frames: sampled tensor [num_samples, ...]
        cur_pos: int, the index of the current frame inside `frames`
    """
    T = video.size(0)
    num_samples = max(1, min(num_samples, T))
    # cur = int(round(current_prop * (T - 1)))
    cur = current_frame

    # initial uniform grid
    idx = torch.linspace(0, T - 1, steps=num_samples).round().long()
    idx[0], idx[-1] = 0, T - 1

    # ensure current included
    if (idx != cur).all():
        pos = (idx - cur).abs().argmin()
        idx[pos] = cur

    # unique and sorted
    idx = torch.unique_consecutive(torch.sort(idx)[0])

    # adjust size if needed
    if idx.numel() > num_samples:
        take = torch.linspace(0, idx.numel() - 1, steps=num_samples).round().long()
        idx = idx[take]
    elif idx.numel() < num_samples:
        # pad if too few (short video case)
        extra = [i for i in range(T) if i not in idx.tolist()]
        idx = torch.sort(torch.cat([idx, torch.tensor(extra[: num_samples - idx.numel()])]))[0]

    frames = video[idx]
    cur_pos = (idx == cur).nonzero(as_tuple=True)[0].item()

    return frames, cur_pos

def process_video_with_spatracker_optimized(video_path, metadata, vggt4track_model, current_frame, config, device, dtype, first_frame_path, temp_frame_dir=None):
    """
    Memory-optimized video processing using SpaTrackerV2 models, following inference.py approach.
    
    Args:
        video_path: Path to world model video
        metadata: Metadata dictionary
        vggt4track_model: VGGT4Track model
        current_frame: Current frame index
        config: Configuration dictionary
        device: Computing device
        dtype: Data type
        first_frame_path: Path to first frame image
        temp_frame_dir: Directory to save extracted frames
    """
    spatracker_config = config.get('spatracker', {})
    fps = spatracker_config.get('fps', 3)  # Use fps=3 like working example
    max_frames = int(spatracker_config.get('max_frames_offline', 80))
    grid_size = int(spatracker_config.get('grid_size', 10))
    vo_points = int(spatracker_config.get('vo_points', 756))
    track_mode = spatracker_config.get('track_mode', 'offline')
    
    # Extract frames from world model video (like VGGT does)
    print(f"Extracting frames from world model video...")
    if temp_frame_dir is None:
        temp_frame_dir = os.path.join(PROJECT_ROOT, 'video_frames')
    predicted_image_paths = extract_frames(video_path, temp_frame_dir)
    
    # Setup camera paths: predicted frames + first frame (like VGGT)
    camera_paths = [first_frame_path]
    camera_paths.extend(predicted_image_paths)
    
    print(f"Loading {len(camera_paths)} frames (predicted + first frame)...")
    
    # Load all frames as images and resize to consistent size
    frames_list = []
    target_size = None
    
    for i, img_path in enumerate(camera_paths):
        img = Image.open(img_path).convert("RGB")
        
        # Set target size from first frame
        if target_size is None:
            target_size = img.size  # (width, height)
            print(f"Target frame size: {target_size[0]}x{target_size[1]}")
        
        # Resize if necessary
        if img.size != target_size:
            print(f"  Resizing frame {i} from {img.size[0]}x{img.size[1]} to {target_size[0]}x{target_size[1]}")
            img = img.resize(target_size, Image.LANCZOS)
        
        img_np = np.array(img)
        frames_list.append(img_np)
    
    # Convert to tensor (T, H, W, 3) -> (T, 3, H, W)
    video_tensor = torch.from_numpy(np.stack(frames_list)).permute(0, 3, 1, 2).float()

    # Resize video to smaller resolution to save memory
    h, w = video_tensor.shape[2:]
    # Use much smaller resolution, similar to inference.py resize logic
    max_size = 336  # Same as inference.py
    scale = min(max_size / h, max_size / w)
    if scale < 1:
        new_h, new_w = int(h * scale), int(w * scale)
        import torchvision.transforms as T
        video_tensor = T.Resize((new_h, new_w))(video_tensor)
        # first_frame_tensor = T.Resize((new_h, new_w))(first_frame_tensor)

    # video_tensor = torch.cat([first_frame_tensor, video_tensor], dim=0)  # Add first frame at the start, (T+1, 3, H, W)
    
    print(f"Video tensor shape after resize: {video_tensor.shape}")
    
    # Process with VGGT4Track model (following inference.py exactly)
    video_tensor = preprocess_image(video_tensor)[None]
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # Use bfloat16 like inference.py
            # Predict poses, depths, and point maps
            predictions = vggt4track_model(video_tensor.cuda() / 255)  # Move to cuda here
            extrinsic = predictions["poses_pred"]  # Camera poses
            intrinsic = predictions["intrs"]       # Camera intrinsics  
            depth_map = predictions["points_map"][..., 2]  # Depth maps
            depth_conf = predictions["unc_metric"]         # Depth confidence
    
    # Convert to numpy immediately and free GPU memory
    extrinsic_np = extrinsic.squeeze().cpu().numpy()
    intrinsic_np = intrinsic.squeeze().cpu().numpy()
    depth_tensor_np = depth_map.squeeze().cpu().numpy()
    unc_metric_np = depth_conf.squeeze().cpu().numpy() > 0.5
    video_tensor = video_tensor.squeeze()

    del predictions, extrinsic, intrinsic, depth_map, depth_conf

    grid_pts = get_points_on_a_grid(grid_size, (h, w), device="cpu")

    query_xyt = torch.cat([torch.zeros_like(grid_pts[:,:,:1]), grid_pts], dim=2)[0].numpy()
    if query_xyt.shape[0] > vo_points:
        query_xyt = query_xyt[np.linspace(0, query_xyt.shape[0]-1, vo_points, dtype=int)]

    if track_mode == "offline":
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    else:
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
    model.spatrack.track_num = vo_points
    model.eval().to(device)

    with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
        (c2w_traj, intrs_out, point_map, conf_depth, track3d_pred, track2d_pred,
         vis_pred, conf_pred, video_out) = model.forward(
            video=video_tensor,  # 与 VGGT 输入一致
            depth=depth_tensor_np,
            intrs=intrinsic_np,
            extrs=extrinsic_np,
            queries=query_xyt,
            fps=fps,
            full_point=False,
            iters_track=4,
            query_no_BA=True,
            fixed_cam=False,
            stage=1,
            unc_metric=unc_metric_np,
            support_frame=len(video_tensor)-1,
            replace_ratio=0.2
        )

    c2w_traj = c2w_traj.float().cpu().numpy() # (T,4,4)
    extrinsics_w2c = np.linalg.inv(c2w_traj)  # Convert to world-to-camera format
    
    # Clear GPU memory immediately
    del video_tensor, track3d_pred, track2d_pred, vis_pred, conf_pred, video_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Prepare results dictionary
    results = {
        "c2w_traj": c2w_traj,
        'extrinsics': extrinsics_w2c,
        'intrinsics': intrs_out.float().cpu().numpy(),
        'depths': point_map[:,2,...].float().cpu().numpy(),
        "depth_conf": conf_depth.float().cpu().numpy()
    }
    
    return results

def process_single_egowalk_segment_with_models(segment_path, vggt4track_model, spatracker_model, current_frame, config, device, dtype, video_path=None):
    """Process a single EgoWalk segment with pre-loaded models."""
    segment_name = os.path.basename(segment_path.rstrip('/'))
    print(f"Processing single segment with SpaTrackerV2: {segment_name}")
    
    try:
        # Construct file paths based on segment name
        current_frame_path = os.path.join(segment_path, f"{segment_name}_current_frame.png")
        first_frame_path = os.path.join(segment_path, f"{segment_name}_first_frame.png") 
        metadata_path = os.path.join(segment_path, f"{segment_name}_metadata.json")
        
        # If video_path not provided, search for it in segment directory
        if video_path is None:
            video_files = glob.glob(os.path.join(segment_path, "*current_frame.mp4"))
            if not video_files:
                print(f"Error: No world model video found in {segment_path}")
                return None
            video_path = video_files[0]

            if os.path.exists(os.path.join(segment_path, "combined_2.mp4")):
                video_path = os.path.join(segment_path, "combined_2.mp4")
        
        # Check if required files exist
        if not all(os.path.exists(path) for path in [current_frame_path, first_frame_path, metadata_path]):
            print(f"Error: Missing required files in {segment_path}")
            return None
        
        # Load metadata to get real displacement
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        real_displacement = metadata["distance_first_to_current"]["distance_meters"]
        print(f"Real displacement: {real_displacement} meters")
        
        # Process video with SpaTrackerV2
        print(f"Processing video with SpaTrackerV2: {os.path.basename(video_path)}")
        temp_frame_dir = os.path.join(PROJECT_ROOT, "pipelines/spatracker/output/video_frames")
        spatracker_results = process_video_with_spatracker_optimized(video_path, metadata, vggt4track_model, current_frame, config, device, dtype, first_frame_path, temp_frame_dir=temp_frame_dir)
        
        # Extract extrinsics (camera poses)
        extrinsics = spatracker_results['c2w_traj']  # Shape: (num_frames, 3, 4) or (num_frames, 4, 4)
        
        print(f"Extrinsics shape: {extrinsics.shape}")

        # #### The direct output extrinsics are in world-to-camera format ####
        # #### We need to convert them to camera-to-world format for correct position extraction ####
        # extrinsics_c2w = []
        # for extr in extrinsics:
        #     if extr.shape == (4, 4):
        #         R = extr[:3, :3]
        #         t = extr[:3, 3]
        #     elif extr.shape == (3, 4):
        #         R = extr[:, :3]
        #         t = extr[:, 3]
        #     else:
        #         print(f"Warning: Unexpected extrinsic shape: {extr.shape}")
        #         continue
        #     R_c2w = R.T
        #     t_c2w = -R_c2w @ t
        #     extr_c2w = np.eye(4)
        #     extr_c2w[:3, :3] = R_c2w
        #     extr_c2w[:3, 3] = t_c2w
        #     extrinsics_c2w.append(extr_c2w)
        # extrinsics_c2w = np.array(extrinsics_c2w)
        # extrinsics = extrinsics_c2w  # Use camera-to-world extrinsics for position extraction
        
        # Handle different extrinsic matrix formats
        if len(extrinsics.shape) == 3 and extrinsics.shape[-1] == 4:
            if extrinsics.shape[-2] == 4:
                # Full 4x4 transformation matrices - extract 3x4 part
                extrinsics_3x4 = extrinsics[:, :3, :]
            else:
                # Already 3x4 matrices
                extrinsics_3x4 = extrinsics
        else:
            print(f"Warning: Unexpected extrinsics shape: {extrinsics.shape}")
            return None
        
        # Extract camera positions from extrinsic matrices
        if len(extrinsics_3x4) < 2:
            print("Error: Need at least 2 frames to compute displacement")
            return None
        
        # Extract camera positions for first and second frames
        current_pos = spatracker_results['current_pos']
        camera1_position = extrinsics_3x4[0, :3, 3] # First frame position
        camera2_position = extrinsics_3x4[current_pos, :3, 3] # Second frame position
        
        # Calculate Euclidean distance between camera positions
        camera_displacement = np.linalg.norm(camera1_position - camera2_position)
        
        print(f"Camera 1 position: {camera1_position}")
        print(f"Camera 2 position: {camera2_position}")
        print(f"Camera displacement: {camera_displacement}")
        
        # Compute scale factor
        if camera_displacement == 0:
            print(f"Warning: Zero camera displacement for segment {segment_name}")
            scale_factor = float('inf')
        else:
            scale_factor = real_displacement / camera_displacement
        
        print(f"Scale factor: {scale_factor}")
        
        # Return results
        result = {
            'segment_name': segment_name,
            'real_displacement': real_displacement,
            'camera_displacement': camera_displacement,
            'scale_factor': scale_factor,
            'extrinsic': extrinsics,
            'current_pos': current_pos,
            'intrinsic': spatracker_results['intrinsics'],
            'depths': spatracker_results['depths'],
            'spatracker_results': spatracker_results
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing segment {segment_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_single_segment(segment_path, config_path="spatracker_pcd_configs.yml"):
    """Process a single EgoWalk segment and compute scale factor using SpaTrackerV2."""
    # Load configuration
    config = load_config(config_path)
    
    # Setup models
    vggt4track_model, spatracker_model, device, dtype = setup_spatracker_models(config)
    
    segment_name = os.path.basename(segment_path.rstrip('/'))
    print(f"Processing single segment with SpaTrackerV2: {segment_name}")
    
    try:
        # Construct file paths based on segment name
        current_frame_path = os.path.join(segment_path, f"{segment_name}_current_frame.png")
        first_frame_path = os.path.join(segment_path, f"{segment_name}_first_frame.png") 
        metadata_path = os.path.join(segment_path, f"{segment_name}_metadata.json")
        
        # Find world model generated video (starts with "Move")
        video_files = glob.glob(os.path.join(segment_path, "Move*.mp4"))
        if not video_files:
            print(f"Error: No world model video found in {segment_path}")
            return None
        video_path = video_files[0]
        
        # Check if required files exist
        if not all(os.path.exists(path) for path in [current_frame_path, first_frame_path, metadata_path]):
            print(f"Error: Missing required files in {segment_path}")
            return None
        
        # Load metadata to get real displacement
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        real_displacement = metadata["distance_first_to_current"]["distance_meters"]
        print(f"Real displacement: {real_displacement} meters")
        current_frame = int(metadata.get("current_frame_index", metadata.get("current_frame", 0)))
        
        # Process video with SpaTrackerV2
        print(f"Processing video with SpaTrackerV2: {os.path.basename(video_path)}")
        temp_frame_dir = os.path.join(PROJECT_ROOT, "pipelines/spatracker/output/video_frames")
        spatracker_results = process_video_with_spatracker_optimized(video_path, metadata, vggt4track_model, current_frame, config, device, dtype, first_frame_path, temp_frame_dir=temp_frame_dir)
        
        # Extract extrinsics (camera poses)
        extrinsics = spatracker_results['extrinsics']  # Shape: (num_frames, 3, 4) or (num_frames, 4, 4)
        
        print(f"Extrinsics shape: {extrinsics.shape}")
        
        # Handle different extrinsic matrix formats
        if len(extrinsics.shape) == 3 and extrinsics.shape[-1] == 4:
            if extrinsics.shape[-2] == 4:
                # Full 4x4 transformation matrices - extract 3x4 part
                extrinsics_3x4 = extrinsics[:, :3, :]
            else:
                # Already 3x4 matrices
                extrinsics_3x4 = extrinsics
        else:
            print(f"Warning: Unexpected extrinsics shape: {extrinsics.shape}")
            return None
        
        # Extract camera positions from extrinsic matrices
        # Extrinsics are in the format [R|t] where t is the camera position
        # For camera-to-world transformation: position = -R^T @ t
        
        if len(extrinsics_3x4) < 2:
            print("Error: Need at least 2 frames to compute displacement")
            return None
        
        # Extract camera positions for first and second frames
        # Method 1: Direct translation extraction (assuming world-to-camera format)
        camera1_position = extrinsics_3x4[0, :3, 3]  # First frame position
        camera2_position = extrinsics_3x4[1, :3, 3]  # Second frame position
        
        # Calculate Euclidean distance between camera positions
        camera_displacement = np.linalg.norm(camera1_position - camera2_position)
        
        print(f"Camera 1 position: {camera1_position}")
        print(f"Camera 2 position: {camera2_position}")
        print(f"Camera displacement: {camera_displacement}")
        
        # Compute scale factor
        if camera_displacement == 0:
            print(f"Warning: Zero camera displacement for segment {segment_name}")
            scale_factor = float('inf')
        else:
            scale_factor = real_displacement / camera_displacement
        
        print(f"Scale factor: {scale_factor}")
        
        # Return results
        result = {
            'segment_name': segment_name,
            'real_displacement': real_displacement,
            'camera_displacement': camera_displacement,
            'scale_factor': scale_factor,
            'extrinsic': extrinsics,
            'intrinsic': spatracker_results['intrinsics'],
            'depths': spatracker_results['depths'],
            'spatracker_results': spatracker_results
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing segment {segment_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_egowalk_segments(config_path="spatracker_pcd_configs.yml", segments_dir=None, video_processing_data=None):
    """Process all EgoWalk segments and compute scale factors using SpaTrackerV2."""
    # Load configuration
    config = load_config(config_path)
    
    if segments_dir is None:
        segments_dir = config['paths']['segments_dir']
    
    # Setup models ONCE for all segments
    vggt4track_model, spatracker_model, device, dtype = setup_spatracker_models(config)
    
    # Find all segment directories
    segment_dirs = glob.glob(os.path.join(segments_dir, "*/"))
    print(f"Found {len(segment_dirs)} EgoWalk segments to process with SpaTrackerV2")
    
    all_results = []
    
    for segment_dir in tqdm(segment_dirs, desc="Processing segments"):
        print(f"\nProcessing segment: {os.path.basename(segment_dir.rstrip('/'))}")

        folder_name = os.path.basename(segment_dir.rstrip('/'))
        # first_frames = video_processing_data[folder_name]['first_fixed_frames']
        # total_frames = video_processing_data[folder_name]['combined_frames']
        # current_prop = first_frames / total_frames if total_frames > 0 else 0.0

        current_frame = int(video_processing_data[folder_name]['first_current_fixed_frames'])
        
        try:
            # Process single segment with shared models
            result = process_single_egowalk_segment_with_models(segment_dir, vggt4track_model, spatracker_model, current_frame, config, device, dtype)
            
            if result:
                all_results.append(result)
                print(f"✓ Successfully processed {result['segment_name']}")
                print(f"  Scale factor: {result['scale_factor']:.4f}")
                
                # Clear GPU cache after each segment to prevent memory buildup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"✗ Failed to process {segment_dir}")
                
        except Exception as e:
            print(f"Error processing segment {segment_dir}: {e}")
            # Clear GPU cache on error to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    return all_results

if __name__ == "__main__":
    # Process all EgoWalk segments
    results = process_egowalk_segments()
    print("Processing complete!")
    
    # Print summary statistics
    if results:
        scale_factors = [r['scale_factor'] for r in results if r['scale_factor'] != float('inf')]
        if scale_factors:
            print(f"\nSummary Statistics:")
            print(f"Number of processed segments: {len(results)}")
            print(f"Valid scale factors: {len(scale_factors)}")
            print(f"Mean scale factor: {np.mean(scale_factors):.4f}")
            print(f"Median scale factor: {np.median(scale_factors):.4f}")
            print(f"Std scale factor: {np.std(scale_factors):.4f}")
            print(f"Min scale factor: {np.min(scale_factors):.4f}")
            print(f"Max scale factor: {np.max(scale_factors):.4f}")
        
        # Save results to JSON file
        output_file = os.path.join(PROJECT_ROOT, "pipelines/spatracker/output/scale_factor_results.json")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = []
        for result in results:
            result_copy = result.copy()
            # Convert numpy arrays to lists, skip complex nested structures
            for key in ['extrinsic', 'intrinsic', 'depths']:
                if key in result_copy and isinstance(result_copy[key], np.ndarray):
                    result_copy[key] = result_copy[key].tolist()
            # Remove complex objects that can't be serialized
            if 'spatracker_results' in result_copy:
                del result_copy['spatracker_results']
            results_serializable.append(result_copy)
        
        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"Results saved to: {output_file}")
    else:
        print("No segments were successfully processed.")
