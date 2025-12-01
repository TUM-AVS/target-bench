"""
Modified to process EgoWalk_samples dataset instead of pkl files.

Key changes:
1. Replaced process_pkl_files() with process_egowalk_segments()
2. Added process_single_egowalk_segment() for testing individual segments
3. Data loading now reads from:
   - {segment_name}_current_frame.png (current frame)
   - {segment_name}_first_frame.png (reference frame) 
   - {segment_name}_metadata.json (contains real displacement)
   - Move*.mp4 (world model generated video)
4. Real displacement extracted from metadata.json: ["distance_first_to_current"]["distance_meters"]
5. Camera positions compared between reference frame and first predicted frame from world model
6. Results saved to JSON file with summary statistics
"""

import torch
import numpy as np
import open3d as o3d
import os
import pickle
import glob
import cv2
import json
import yaml
import struct
from pathlib import Path
import pandas as pd
import sys
# Define project root
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../../"))
sys.path.append(os.path.join(PROJECT_ROOT, 'models/vggt'))

from tqdm.auto import tqdm
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
# Sky segmentation imports
try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

import requests
import copy

def load_config(config_path="vggt_pcd_configs.yml"):
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
            'model': {'confidence_threshold': 0.5, 'apply_sky_mask': True},
            'post_processing': {
                'fix_coordinates': {'enabled': True},
                'filter_points': {'enabled': True, 'min_z': -2.0, 'max_z': 0.5},
                'scale_coordinates': {'enabled': True, 'scale_factor': 1.1}
            },
            'paths': {  
                'pkl_dir': os.path.join(PROJECT_ROOT, "dataset/navsim_logs/mini"),
                'images_base_path': os.path.join(PROJECT_ROOT, "dataset/sensor_blobs/mini"),
                'output_base_path': os.path.join(PROJECT_ROOT, "dataset/sensor_blobs/mini"),
                'camera_params_json': os.path.join(PROJECT_ROOT, "camera_para.json")
            },
            'cameras': {'camera_order': ['CAM_F0', 'CAM_L0', 'CAM_R0']}
        }

def fix_pcd_coordinates(pcd_file_path):
    """
    Fix coordinate sequence in a PCD file from (x, z, y) to (x, y, z).
    
    Args:
        pcd_file_path (str): Path to PCD file to fix
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the point cloud
        pcd = o3d.io.read_point_cloud(pcd_file_path)
        
        if len(pcd.points) == 0:
            print(f"Warning: {pcd_file_path} contains no points")
            return False
        
        # Get the points as numpy array
        points = np.asarray(pcd.points)
        
        # Current sequence: x, z, y -> Target sequence: y, -x, z
        # Transform: x -> y, z -> -x, y -> z
        # More efficient: create the transformation in one operation
        corrected_points = np.column_stack([points[:, 2], -points[:, 0], points[:, 1]])
        
        # Update the point cloud with corrected coordinates
        pcd.points = o3d.utility.Vector3dVector(corrected_points)
        
        # Save the corrected point cloud
        success = o3d.io.write_point_cloud(pcd_file_path, pcd)
        
        if success:
            print(f"✓ Fixed coordinates in {os.path.basename(pcd_file_path)}")
            return True
        else:
            print(f"✗ Failed to write {pcd_file_path}")
            return False
            
    except Exception as e:
        print(f"✗ Error fixing coordinates in {pcd_file_path}: {str(e)}")
        return False

def read_pcd_header(filepath: str):
    """Read PCD file header and return header info and data start position."""
    header = {}
    
    with open(filepath, 'rb') as f:
        while True:
            pos = f.tell()
            line = f.readline().decode('ascii').strip()
            
            if line.startswith('DATA'):
                header['data_type'] = line.split()[1]
                return header, f.tell()
            
            if line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]
                if key in ['VERSION', 'WIDTH', 'HEIGHT', 'POINTS']:
                    header[key] = parts[1]
                elif key in ['FIELDS', 'SIZE', 'TYPE', 'COUNT']:
                    header[key] = parts[1:]
                elif key == 'VIEWPOINT':
                    header[key] = parts[1:]
    
    return header, 0

def update_header_points_count(header_data: bytes, new_point_count: int) -> bytes:
    """Update the POINTS count in the header data."""
    header_str = header_data.decode('ascii')
    lines = header_str.split('\n')
    
    updated_lines = []
    for line in lines:
        if line.startswith('POINTS'):
            updated_lines.append(f'POINTS {new_point_count}')
        else:
            updated_lines.append(line)
    
    return '\n'.join(updated_lines).encode('ascii')

def filter_pcd_file(filepath: str, min_z: float = -2.0, max_z: float = 0.5):
    """
    Remove points from a PCD file where z-coordinate is outside the range [min_z, max_z].
    
    Args:
        filepath: Path to the PCD file
        min_z: Minimum z value (inclusive) - points below this will be removed
        max_z: Maximum z value (inclusive) - points above this will be removed
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read header
        header, data_start = read_pcd_header(filepath)
        
        # Check if it's the expected format (x y z rgb)
        if header.get('FIELDS') != ['x', 'y', 'z', 'rgb']:
            print(f"Warning: Unexpected fields in {filepath}: {header.get('FIELDS')}")
            return False
        
        if header.get('data_type') != 'binary':
            print(f"Warning: File {filepath} is not binary format")
            return False
        
        # Calculate number of points and bytes per point
        num_points = int(header['POINTS'])
        bytes_per_point = sum(int(size) for size in header['SIZE'])
        
        # Read the original file
        with open(filepath, 'rb') as f:
            # Read header as text
            f.seek(0)
            header_data = f.read(data_start)
            
            # Read binary point cloud data
            f.seek(data_start)
            binary_data = f.read()
        
        # Process the binary data and filter points
        filtered_data = bytearray()
        kept_points = 0
        removed_points = 0
        
        for i in range(num_points):
            # Read one point (16 bytes: 4 floats of 4 bytes each)
            point_start = i * bytes_per_point
            point_data = binary_data[point_start:point_start + bytes_per_point]
            
            # Unpack the point (x, y, z, rgb as floats)
            x, y, z, rgb = struct.unpack('<ffff', point_data)
            
            # Filter condition: keep points where min_z <= z <= max_z
            if min_z <= z <= max_z:
                filtered_data.extend(point_data)
                kept_points += 1
            else:
                removed_points += 1
        
        # Update header with new point count
        updated_header = update_header_points_count(header_data, kept_points)
        
        # Write the filtered data back to the file
        with open(filepath, 'wb') as f:
            f.write(updated_header)
            f.write(filtered_data)
        
        print(f"✓ Filtered {os.path.basename(filepath)}: kept {kept_points}, removed {removed_points} points (z range: [{min_z}, {max_z}])")
        return True
        
    except Exception as e:
        print(f"✗ Error filtering {filepath}: {str(e)}")
        return False

def scale_pcd_file(filepath: str, scale_factor: float = 1.1):
    """
    Scale x, y, z coordinates in a PCD file by the given factor.
    
    Args:
        filepath: Path to the PCD file
        scale_factor: Factor to multiply x, y, z coordinates by
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read header
        header, data_start = read_pcd_header(filepath)
        
        # Check if it's the expected format (x y z rgb)
        if header.get('FIELDS') != ['x', 'y', 'z', 'rgb']:
            print(f"Warning: Unexpected fields in {filepath}: {header.get('FIELDS')}")
            return False
        
        if header.get('data_type') != 'binary':
            print(f"Warning: File {filepath} is not binary format")
            return False
        
        # Calculate number of points and bytes per point
        num_points = int(header['POINTS'])
        bytes_per_point = sum(int(size) for size in header['SIZE'])
        
        # Read the original file
        with open(filepath, 'rb') as f:
            # Read header as text
            f.seek(0)
            header_data = f.read(data_start)
            
            # Read binary point cloud data
            f.seek(data_start)
            binary_data = f.read()
        
        # Process the binary data
        scaled_data = bytearray()
        
        for i in range(num_points):
            # Read one point (16 bytes: 4 floats of 4 bytes each)
            point_start = i * bytes_per_point
            point_data = binary_data[point_start:point_start + bytes_per_point]
            
            # Unpack the point (x, y, z, rgb as floats)
            x, y, z, rgb = struct.unpack('<ffff', point_data)
            
            # Scale x, y, z coordinates
            x_scaled = x * scale_factor
            y_scaled = y * scale_factor
            z_scaled = z * scale_factor
            
            # Pack the scaled point back to binary
            scaled_point = struct.pack('<ffff', x_scaled, y_scaled, z_scaled, rgb)
            scaled_data.extend(scaled_point)
        
        # Write the scaled data back to the file
        with open(filepath, 'wb') as f:
            f.write(header_data)
            f.write(scaled_data)
        
        print(f"✓ Scaled {os.path.basename(filepath)} by factor {scale_factor}")
        return True
        
    except Exception as e:
        print(f"✗ Error scaling {filepath}: {str(e)}")
        return False

def apply_pcd_post_processing(pcd_file_path, config, scale_factor=None):
    """
    Apply post-processing steps to a PCD file in the specified order.
    
    Args:
        pcd_file_path: Path to the PCD file
        config: Configuration dictionary
    """
    print(f"Applying post-processing to {os.path.basename(pcd_file_path)}")
    
    post_proc_config = config.get('post_processing', {})
    
    # Step 1: Fix coordinates (x,z,y) -> (x,y,z)
    if post_proc_config.get('fix_coordinates', {}).get('enabled', False):
        print("  1. Fixing coordinate sequence...")
        fix_pcd_coordinates(pcd_file_path)

    # Step 2: Scale coordinates
    print(f"  3. Scaling coordinates by factor {scale_factor}...")
    scale_pcd_file(pcd_file_path, scale_factor)
    
    # Step 3: Filter points by z-coordinate range
    if post_proc_config.get('filter_points', {}).get('enabled', False):
        min_z = post_proc_config.get('filter_points', {}).get('min_z', -30.0)
        max_z = post_proc_config.get('filter_points', {}).get('max_z', 6)
        print(f"  2. Filtering points (z range: [{min_z}, {max_z}])...")
        filter_pcd_file(pcd_file_path, min_z, max_z)

def setup_model():
    """Initialize the VGGT model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    print("Loading VGGT model...")
    model = VGGT()
    model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, "models/vggt/VGGT-1B/model.pt"), map_location=device))

    model = model.to(device)
    model.eval()

    return model, device, dtype

def extract_camera_paths(frame_data, base_path="dataset", camera_order=None):
    """Extract camera image paths from frame data in the correct order."""
    if camera_order is None:
        camera_order = ['CAM_F0', 'CAM_L0', 'CAM_R0']  # Default to 3 cameras
    
    camera_paths = []
    cams = frame_data.get('cams', {})
    
    for cam_key in camera_order:
        if cam_key in cams:
            cam_data = cams[cam_key]
            data_path = cam_data.get('data_path', '')
            if data_path:
                # Construct full path
                full_path = os.path.join(base_path, data_path)
                if os.path.exists(full_path):
                    camera_paths.append(full_path)
                else:
                    print(f"Warning: Image file not found: {full_path}")
                    return None  # Return None if any camera is missing
            else:
                print(f"Warning: No data_path for camera {cam_key}")
                return None
        else:
            print(f"Warning: Camera {cam_key} not found in frame data")
            return None
    
    if len(camera_paths) != len(camera_order):
        print(f"Warning: Expected {len(camera_order)} cameras, got {len(camera_paths)}")
        return None
        
    return camera_paths

def download_file_from_url(url, filename):
    """Downloads a file from a Hugging Face model repo, handling redirects."""
    try:
        # Get the redirect URL
        response = requests.get(url, allow_redirects=False)
        response.raise_for_status()  # Raise HTTPError for bad requests (4xx or 5xx)

        if response.status_code == 302:  # Expecting a redirect
            redirect_url = response.headers["Location"]
            response = requests.get(redirect_url, stream=True)
            response.raise_for_status()
        else:
            print(f"Unexpected status code: {response.status_code}")
            return

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} successfully.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")

def run_skyseg(onnx_session, input_size, image):
    """
    Runs sky segmentation inference using ONNX model.

    Args:
        onnx_session: ONNX runtime session
        input_size: Target size for model input (width, height)
        image: Input image in BGR format

    Returns:
        np.ndarray: Segmentation mask
    """

    # Pre process:Resize, BGR->RGB, Transpose, PyTorch standardization, float32 cast
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    # Post process
    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype("uint8")

    return onnx_result

def segment_sky(image_path, onnx_session, mask_filename=None):
    """
    Segments sky from an image using an ONNX model.
    Thanks for the great model provided by https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing

    Args:
        image_path: Path to input image
        onnx_session: ONNX runtime session with loaded model
        mask_filename: Path to save the output mask

    Returns:
        np.ndarray: Binary mask where 255 indicates non-sky regions
    """

    assert mask_filename is not None
    image = cv2.imread(image_path)

    result_map = run_skyseg(onnx_session, [320, 320], image)
    # resize the result_map to the original image size
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    # Fix: Invert the mask so that 255 = non-sky, 0 = sky
    # The model outputs low values for sky, high values for non-sky
    output_mask = np.zeros_like(result_map_original)
    output_mask[result_map_original < 32] = 255  # Use threshold of 32

    #os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    #cv2.imwrite(mask_filename, output_mask)
    return output_mask

def apply_sky_segmentation(conf: np.ndarray, image_folder: str, image_names: list) -> np.ndarray:
    """
    Apply sky segmentation to confidence scores.

    Args:
        conf (np.ndarray): Confidence scores with shape (S, H, W)
        image_folder (str): Path to the folder containing input images
        image_names (list): List of image file paths

    Returns:
        np.ndarray: Updated confidence scores with sky regions masked out
    """
    S, H, W = conf.shape
    sky_masks_dir = image_folder.rstrip("/") + "_sky_masks"
    os.makedirs(sky_masks_dir, exist_ok=True)

    # Download skyseg.onnx if it doesn't exist
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    sky_mask_list = []

    print("Generating sky masks...")
    for i, image_path in enumerate(tqdm(image_names[:S])):  # Limit to the number of images in the batch
        image_name = os.path.basename(image_path)
        mask_filepath = os.path.join(sky_masks_dir, image_name)

        if os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)

        # Resize mask to match H×W if needed
        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H))

        sky_mask_list.append(sky_mask)

    # Convert list to numpy array with shape S×H×W
    sky_mask_array = np.array(sky_mask_list)
    # Apply sky mask to confidence scores (sky regions have low values, non-sky regions have high values)
    # We want to keep non-sky regions, so we use the mask as is
    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
    conf = conf * sky_mask_binary

    print("Sky segmentation applied successfully")
    return conf

def load_camera_parameters(json_path=None, config=None):
    """Load camera parameters from JSON file."""
    if json_path is None and config is not None:
        json_path = config['paths']['camera_params_json']
    elif json_path is None:
        json_path = os.path.join(PROJECT_ROOT, "camera_para.json")
    
    with open(json_path, 'r') as f:
        camera_params = json.load(f)
    return camera_params

def convert_camera_params_to_matrices(camera_params, camera_names):
    """
    Convert camera parameters from JSON format to extrinsic and intrinsic matrices.
    
    Args:
        camera_params: Dictionary loaded from camera_para.json
        camera_names: List of camera names in sequence
        
    Returns:
        extrinsics: numpy array of shape (S, 3, 4)
        intrinsics: numpy array of shape (S, 3, 3)
    """
    S = len(camera_names)
    extrinsics = np.zeros((S, 3, 4))
    intrinsics = np.zeros((S, 3, 3))
    
    for i, cam_name in enumerate(camera_names):
        # Get intrinsic matrix
        intrinsic_list = camera_params["intrinsics"]['cam_f0'] # intrinsics are the same for all cameras
        intrinsics[i] = np.array(intrinsic_list)
        
        # Get extrinsic matrix (sensor2lidar transformation)
        rotation = np.array(camera_params["extrinsics"][cam_name]["sensor2lidar_rotation"])
        translation = np.array(camera_params["extrinsics"][cam_name]["sensor2lidar_translation"])
        
        # Construct 3x4 extrinsic matrix [R|t]
        extrinsics[i, :3, :3] = rotation
        extrinsics[i, :3, 3] = translation
    
    return extrinsics, intrinsics

def save_vggt_pcd(predictions, output_path, confidence_threshold=0.5, apply_sky_mask=True, camera_paths=None, config=None, scale_factor=None):
    """Save VGGT predictions to PCD format using depth map unprojection."""
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Debug: Print what keys are available in predictions
    print(f"Available prediction keys: {list(predictions.keys())}")
    
    # Check if depth predictions are available
    if True:#"depth" not in predictions or "depth_conf" not in predictions:
        print("Warning: Depth predictions not available. Falling back to world_points method.")
        # Fall back to the original method using world_points
        if "world_points" not in predictions or "world_points_conf" not in predictions:
            print("Error: Neither depth nor world_points predictions are available!")
            return
        
        # Use original method with world_points
        world_points = predictions["world_points"]  # Shape: [B, S, H, W, 3]
        world_points_conf = predictions["world_points_conf"]  # Shape: [B, S, H, W]
        
        world_points = world_points[:, :-1, :, :, :]  # Remove last camera
        world_points_conf = world_points_conf[:, :-1, :, :]  # Remove last camera

        # Convert to numpy
        world_points_np = world_points.cpu().numpy()
        world_points_conf_np = world_points_conf.cpu().numpy()
        
        B, S, H, W, _ = world_points_np.shape
        print(f"Processing {B} batch(es), {S} camera(s), resolution {H}x{W} using world_points")
        
        # Apply sky segmentation if requested and camera paths are available
        if apply_sky_mask and camera_paths is not None:
            print("Applying sky segmentation...")
            # Get the base directory for sky masks
            base_image_folder = os.path.dirname(camera_paths[0])
            
            # Process each batch
            for b in range(B):
                conf_batch = world_points_conf_np[b]  # Shape: [S, H, W]
                conf_batch = apply_sky_segmentation(conf_batch, base_image_folder, camera_paths)
                world_points_conf_np[b] = conf_batch
        
        # Combine points from all cameras using original method
        all_points = []
        all_conf = []
        
        for b in range(B):
            for s in range(S):  # Process all cameras, not just the first one
                # Get points and confidence for current batch and camera
                points = world_points_np[b, s]  # Shape: [H, W, 3]
                conf = world_points_conf_np[b, s]  # Shape: [H, W]
                
                # Reshape to list of points
                points_flat = points.reshape(-1, 3)  # Shape: [H*W, 3]
                conf_flat = conf.reshape(-1)  # Shape: [H*W]
                
                # Filter by confidence threshold
                valid_mask = conf_flat >= confidence_threshold
                valid_points = points_flat[valid_mask]
                valid_conf = conf_flat[valid_mask]
                
                if len(valid_points) > 0:
                    all_points.append(valid_points)
                    all_conf.append(valid_conf)
                    print(f"Camera {s}: {len(valid_points)}/{len(points_flat)} points above threshold {confidence_threshold}")
        
        if len(all_points) == 0:
            print(f"Warning: No points above confidence threshold for {output_path}")
            return
        
        # Combine all points from all cameras
        combined_points = np.concatenate(all_points, axis=0)
        combined_conf = np.concatenate(all_conf, axis=0)
        
        # Filter out points where y-coordinate < -0.1
        # y_filter_mask = combined_points[:, 1] >=  -0.1
        # combined_points = combined_points[y_filter_mask]
        # combined_conf = combined_conf[y_filter_mask]
        
        print(f"Total combined points: {len(combined_points)} from {len(all_points)} cameras (after y > 0.03 filtering)")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_points)
        
        # Color points by confidence (grayscale)
        if len(combined_conf) > 0:
            conf_normalized = (combined_conf - combined_conf.min()) / (combined_conf.max() - combined_conf.min() + 1e-8)
            colors = np.stack([conf_normalized, conf_normalized, conf_normalized], axis=1)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save PCD file
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved PCD: {output_path} ({len(combined_points)} points)")
        
        # Apply post-processing if config is provided
        if config is not None:
            apply_pcd_post_processing(output_path, config, scale_factor=scale_factor)

        return
    
    # Original depth-based method continues here
    # Extract depth maps and confidence scores
    depth_maps = predictions["depth"]  # Shape: [B, S, H, W, 1]
    depth_conf = predictions["depth_conf"]  # Shape: [B, S, H, W]
    
    # Convert to numpy
    depth_maps_np = depth_maps.cpu().numpy()
    depth_conf_np = depth_conf.cpu().numpy()
    
    # Handle depth maps shape - could be [B, S, H, W] or [B, S, H, W, 1]
    if len(depth_maps_np.shape) == 5:
        B, S, H, W, _ = depth_maps_np.shape
        print(f"Processing {B} batch(es), {S} camera(s), resolution {H}x{W} using depth maps (5D)")
    else:
        B, S, H, W = depth_maps_np.shape
        print(f"Processing {B} batch(es), {S} camera(s), resolution {H}x{W} using depth maps (4D)")
    
    # Define camera names sequence as in line 333
    camera_names = ['cam_f0', 'cam_l0', 'cam_l1', 'cam_l2', 'cam_r0', 'cam_r1', 'cam_r2', 'cam_b0']
    
    # Load camera parameters
    camera_params = load_camera_parameters(config=config)
    extrinsics_np, intrinsics_np = convert_camera_params_to_matrices(camera_params, camera_names)
    
    # Convert to torch tensors with proper dimensions for unproject_depth_map_to_point_map
    # The function expects extrinsic: (S, 3, 4) and intrinsic: (S, 3, 3)
    # But we need to add batch dimension to match our processing: (1, S, 3, 4) and (1, S, 3, 3)
    extrinsics_torch = torch.from_numpy(extrinsics_np).float().unsqueeze(0)  # Add batch dimension: (1, S, 3, 4)
    intrinsics_torch = torch.from_numpy(intrinsics_np).float().unsqueeze(0)   # Add batch dimension: (1, S, 3, 3)
    
    # Apply sky segmentation if requested and camera paths are available
    if apply_sky_mask and camera_paths is not None:
        print("Applying sky segmentation...")
        # Get the base directory for sky masks
        base_image_folder = os.path.dirname(camera_paths[0])
        
        # Process each batch
        for b in range(B):
            conf_batch = depth_conf_np[b]  # Shape: [S, H, W]
            conf_batch = apply_sky_segmentation(conf_batch, base_image_folder, camera_paths)
            depth_conf_np[b] = conf_batch
    
    # Combine points from all cameras using unprojection
    all_points = []
    all_conf = []
    
    for b in range(B):
        # Get depth maps for current batch
        if len(depth_maps_np.shape) == 5:
            # Remove the last dimension for unproject_depth_map_to_point_map function
            batch_depth_maps = torch.from_numpy(depth_maps_np[b, :, :, :, 0]).float()  # Shape: [S, H, W]
        else:
            batch_depth_maps = torch.from_numpy(depth_maps_np[b]).float()  # Shape: [S, H, W]
        
        batch_depth_conf = depth_conf_np[b]  # Shape: [S, H, W]
        
        # Unproject depth maps to world coordinates using camera parameters
        world_points = unproject_depth_map_to_point_map(
            batch_depth_maps.unsqueeze(-1), # (S, H, W, 1)
            extrinsics_torch.squeeze(0),  # (S, 3, 4)
            intrinsics_torch.squeeze(0)   # (S, 3, 3)
        )  # Shape: [S, H, W, 3]
        
        # world_points is already a numpy array (unproject_depth_map_to_point_map always returns numpy)
        world_points_np = world_points
        
        # Process each camera
        for s in range(S):
            # Get points and confidence for current camera
            points = world_points_np[s]  # Shape: [H, W, 3]
            conf = batch_depth_conf[s]  # Shape: [H, W]
            
            # Reshape to list of points
            points_flat = points.reshape(-1, 3)  # Shape: [H*W, 3]
            conf_flat = conf.reshape(-1)  # Shape: [H*W]
            
            # Filter by confidence threshold
            valid_mask = conf_flat >= confidence_threshold
            valid_points = points_flat[valid_mask]
            valid_conf = conf_flat[valid_mask]
            
            if len(valid_points) > 0:
                all_points.append(valid_points)
                all_conf.append(valid_conf)
                camera_name = camera_names[s] if s < len(camera_names) else f"camera_{s}"
                print(f"Camera {camera_name}: {len(valid_points)}/{len(points_flat)} points above threshold {confidence_threshold}")
    
    if len(all_points) == 0:
        print(f"Warning: No points above confidence threshold for {output_path}")
        return
    
    # Combine all points from all cameras
    combined_points = np.concatenate(all_points, axis=0)
    combined_conf = np.concatenate(all_conf, axis=0)
    
    # Filter out points where y-coordinate > 0.03
    y_filter_mask = combined_points[:, 1] <= 0.03
    combined_points = combined_points[y_filter_mask]
    combined_conf = combined_conf[y_filter_mask]
    
    print(f"Total combined points: {len(combined_points)} from {len(all_points)} cameras (after y > 0.03 filtering)")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    
    # Color points by confidence (grayscale)
    if len(combined_conf) > 0:
        conf_normalized = (combined_conf - combined_conf.min()) / (combined_conf.max() - combined_conf.min() + 1e-8)
        colors = np.stack([conf_normalized, conf_normalized, conf_normalized], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save PCD file
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved PCD: {output_path} ({len(combined_points)} points)")
    
    # Apply post-processing if config is provided
    if config is not None:
        apply_pcd_post_processing(output_path, config)

def extract_frames(video_path, output_dir, num_frames=9):
    """
    Extract evenly spaced frames from a video.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract (default: 9)
        
    Returns:
        List of paths to extracted frame images
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return []

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Ensure output directory exists
    output_frames_dir = os.path.join(output_dir, os.path.basename(video_path).split('.')[0])
    os.makedirs(output_frames_dir, exist_ok=True)

    # Calculate evenly spaced frame indices
    # We want to sample across the entire video, including first and last frame
    if total_frames < num_frames:
        print(f"Warning: Video has only {total_frames} frames, but {num_frames} requested. Extracting all frames.")
        frame_indices = list(range(total_frames))
    else:
        # Evenly space num_frames across the video
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames_list = []

    # Extract frames at calculated indices
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Calculate the time in seconds for this frame
            time_sec = frame_idx / fps if fps > 0 else 0
            filename = os.path.join(output_frames_dir, f"frame_{i:02d}_idx{frame_idx:04d}_t{time_sec:.2f}s.jpg")
            frames_list.append(filename)
            cv2.imwrite(filename, frame)
            print(f"  Saved frame {i+1}/{len(frame_indices)}: {os.path.basename(filename)}")
        else:
            print(f"  Warning: Could not read frame at index {frame_idx}")

    cap.release()
    return frames_list

def process_egowalk_segments(config_path="vggt_pcd_configs.yml", segments_dir=None):
    """Process all EgoWalk segments and compute scale factors using configuration."""
    # Load configuration
    config = load_config(config_path)
    
    if segments_dir is None:
        segments_dir = os.path.join(PROJECT_ROOT, "dataset/Target_samples")
    
    # Setup model
    model, device, dtype = setup_model()
    
    # Find all segment directories
    segment_dirs = glob.glob(os.path.join(segments_dir, "*/"))
    print(f"Found {len(segment_dirs)} EgoWalk segments to process")
    
    all_results = []
    
    for segment_dir in tqdm(segment_dirs, desc="Processing segments"):
        print(f"\nProcessing segment: {os.path.basename(segment_dir.rstrip('/'))}")
        
        try:
            # Extract segment name from directory
            segment_name = os.path.basename(segment_dir.rstrip('/'))
            
            # Construct file paths based on segment name
            current_frame_path = os.path.join(segment_dir, f"{segment_name}_current_frame.png")
            first_frame_path = os.path.join(segment_dir, f"{segment_name}_first_frame.png")
            metadata_path = os.path.join(segment_dir, f"{segment_name}_metadata.json")
            
            # Find world model generated video (starts with "Move")
            video_files = glob.glob(os.path.join(segment_dir, "Move*.mp4"))
            if not video_files:
                print(f"Warning: No world model video found in {segment_dir}")
                continue
            video_path = video_files[0]  # Take the first one if multiple exist
            
            # Check if required files exist
            if not all(os.path.exists(path) for path in [current_frame_path, first_frame_path, metadata_path]):
                print(f"Warning: Missing required files in {segment_dir}")
                continue
            
            # Load metadata to get real displacement
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            real_displacement = metadata["distance_first_to_current"]["distance_meters"]
            print(f"Real displacement: {real_displacement} meters")
            
            # Setup camera paths: current frame, reference frame, and world model predicted frames
            camera_paths = []
            
            # Extract frames from world model generated video
            print(f"Extracting frames from world model video: {os.path.basename(video_path)}")
            temp_frame_dir = os.path.join(PROJECT_ROOT, "pipelines/vggt/output/video_frames")
            os.makedirs(temp_frame_dir, exist_ok=True)
            predicted_image_paths = extract_frames(video_path, temp_frame_dir)
            camera_paths.extend(predicted_image_paths)
            camera_paths.append(first_frame_path)

            # Load and preprocess images
            images = load_and_preprocess_images(camera_paths).to(device)
            
            # Run VGGT inference
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = model(images)
                
                    images = images[None]  # add batch dimension
                    aggregated_tokens_list, ps_idx = model.aggregator(images)
                            
                # Predict Cameras
                pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
                
                # Extract camera positions from extrinsic matrices (last column, first 3 elements)
                # First camera: reference frame (first_frame.png)
                # Second camera: first predicted frame from world model
                camera1_position = extrinsic[0, 0, :3, 3]  # Reference frame position
                camera2_position = extrinsic[0, -1, :3, 3]  # First predicted frame position
                
                # Calculate Euclidean distance between camera positions
                camera_displacement = torch.norm(camera1_position - camera2_position).item()

            # Compute scale factor
            if camera_displacement == 0:
                print(f"Warning: Zero camera displacement for segment {segment_name}")
                scale_factor = float('inf')
            else:
                scale_factor = real_displacement / camera_displacement

            print(f"Camera displacement: {camera_displacement}")
            print(f"Scale factor: {scale_factor}")
            
            # Store results
            result = {
                'segment_name': segment_name,
                'real_displacement': real_displacement,
                'camera_displacement': camera_displacement,
                'scale_factor': scale_factor,
                'extrinsic': extrinsic.cpu().numpy() if torch.is_tensor(extrinsic) else extrinsic,
                'intrinsic': intrinsic.cpu().numpy() if torch.is_tensor(intrinsic) else intrinsic
            }
            all_results.append(result)
            
        except Exception as e:
            print(f"Error processing segment {segment_dir}: {e}")
            continue
    
    return all_results

def process_single_segment(segment_path, config_path="vggt_pcd_configs.yml", video_path=None):
    """
    Process a single EgoWalk segment and compute scale factor.
    
    Args:
        segment_path: Path to the segment directory containing ground truth files
        config_path: Path to VGGT configuration file
        video_path: Optional path to world model video. If None, will search for Move*.mp4
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup model
    model, device, dtype = setup_model()
    
    segment_name = os.path.basename(segment_path.rstrip('/'))
    print(f"Processing single segment: {segment_name}")
    
    try:
        # Construct file paths based on segment name
        current_frame_path = os.path.join(segment_path, f"{segment_name}_current_frame.png")
        first_frame_path = os.path.join(segment_path, f"{segment_name}_first_frame.png")
        metadata_path = os.path.join(segment_path, f"{segment_name}_metadata.json")
        
        # Get video path - either provided or search for Move*.mp4
        if video_path is None:
            # Find world model generated video (starts with "Move")
            video_files = glob.glob(os.path.join(segment_path, "Move*.mp4"))
            if not video_files:
                print(f"Error: No world model video found in {segment_path}")
                return None
            video_path = video_files[0]
        else:
            # Use provided video path
            if not os.path.exists(video_path):
                print(f"Error: Provided video path does not exist: {video_path}")
                return None
        
        # Check if required files exist
        if not all(os.path.exists(path) for path in [current_frame_path, first_frame_path, metadata_path]):
            print(f"Error: Missing required files in {segment_path}")
            return None
        
        # Load metadata to get real displacement
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        real_displacement = metadata["distance_first_to_current"]["distance_meters"]
        print(f"Real displacement: {real_displacement} meters")
        
        # Setup camera paths: current frame, reference frame, and world model predicted frames
        camera_paths = [first_frame_path]
        
        # Extract frames from world model generated video
        print(f"Extracting frames from world model video: {os.path.basename(video_path)}")
        temp_frame_dir = os.path.join(PROJECT_ROOT, "pipelines/vggt/output/video_frames")
        os.makedirs(temp_frame_dir, exist_ok=True)
        predicted_image_paths = extract_frames(video_path, temp_frame_dir)
        camera_paths.extend(predicted_image_paths)
        
        print(f"Processing segment {segment_name} with {len(camera_paths)} images")
        
        # Load and preprocess images
        images = load_and_preprocess_images(camera_paths).to(device)
        
        # Run VGGT inference
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
            
                images = images[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = model.aggregator(images)
                        
            # Predict Cameras
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            
            # Extract camera positions from extrinsic matrices (last column, first 3 elements)
            # First camera: reference frame (first_frame.png)
            # Second camera: first predicted frame from world model
            camera1_position = extrinsic[0, 0, :3, 3]  # Reference frame position
            camera2_position = extrinsic[0, 1, :3, 3]  # First predicted frame position
            
            # Calculate Euclidean distance between camera positions
            camera_displacement = torch.norm(camera1_position - camera2_position).item()

        # Compute scale factor
        if camera_displacement == 0:
            print(f"Warning: Zero camera displacement for segment {segment_name}")
            scale_factor = float('inf')
        else:
            scale_factor = real_displacement / camera_displacement

        print(f"Camera displacement: {camera_displacement}")
        print(f"Scale factor: {scale_factor}")
        
        # Return results
        result = {
            'segment_name': segment_name,
            'real_displacement': real_displacement,
            'camera_displacement': camera_displacement,
            'scale_factor': scale_factor,
            'extrinsic': extrinsic.cpu().numpy() if torch.is_tensor(extrinsic) else extrinsic,
            'intrinsic': intrinsic.cpu().numpy() if torch.is_tensor(intrinsic) else intrinsic
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing segment {segment_path}: {e}")
        return None

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
        output_file = os.path.join(PROJECT_ROOT, "pipelines/vggt/output/scale_factor_results.json")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = []
        for result in results:
            result_copy = result.copy()
            if 'extrinsic' in result_copy:
                result_copy['extrinsic'] = result_copy['extrinsic'].tolist()
            if 'intrinsic' in result_copy:
                result_copy['intrinsic'] = result_copy['intrinsic'].tolist()
            results_serializable.append(result_copy)
        
        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"Results saved to: {output_file}")
    else:
        print("No segments were successfully processed.")
