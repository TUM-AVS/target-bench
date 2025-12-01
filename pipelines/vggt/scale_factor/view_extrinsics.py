import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import pickle
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT()
#_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.load("/home/alienware3/navsim_workspace/vggt_code/ckpt/model.pt", map_location=device))

# Load and preprocess example images (replace with your own image paths)
#image_names = ["/home/alienware3/navsim_workspace/scene/images/0a93cb2410125631.jpg"]  
# image_names = [
#     '/home/alienware3/navsim_workspace/dataset/sensor_blobs/mini/2021.06.14.18.42.45_veh-12_03445_03902/CAM_F0/d0da7ee333215856.jpg',
#     '/home/alienware3/navsim_workspace/dataset/sensor_blobs/mini/2021.06.14.18.42.45_veh-12_03445_03902/CAM_L0/11dd2407c1505d23.jpg',
#     '/home/alienware3/navsim_workspace/dataset/sensor_blobs/mini/2021.06.14.18.42.45_veh-12_03445_03902/CAM_L1/d2e3a078034b5568.jpg',
#     '/home/alienware3/navsim_workspace/dataset/sensor_blobs/mini/2021.06.14.18.42.45_veh-12_03445_03902/CAM_L2/869cb0ac40c059c2.jpg',
#     '/home/alienware3/navsim_workspace/dataset/sensor_blobs/mini/2021.06.14.18.42.45_veh-12_03445_03902/CAM_R0/433a6d2191cb56a6.jpg',
#     '/home/alienware3/navsim_workspace/dataset/sensor_blobs/mini/2021.06.14.18.42.45_veh-12_03445_03902/CAM_R1/77336a9cc4555c18.jpg',
#     '/home/alienware3/navsim_workspace/dataset/sensor_blobs/mini/2021.06.14.18.42.45_veh-12_03445_03902/CAM_R2/66e683f179375874.jpg',
#     '/home/alienware3/navsim_workspace/dataset/sensor_blobs/mini/2021.06.14.18.42.45_veh-12_03445_03902/CAM_B0/b8371dc0c2d35ef2.jpg',
# ]
base_path = "/home/alienware3/navsim_workspace/dataset/sensor_blobs/mini/"
image_names = []
displacement = 0
displacement_list = []
camera_displacement_list = []
time_step_list = []
historical_image_path_list = []
#load pickle file in /home/alienware3/navsim_workspace/dataset/navsim_logs/2021.06.14.18.42.45_veh-12_03445_03902.pkl
data = pickle.load(open('/home/alienware3/navsim_workspace/dataset/navsim_logs/mini/2021.06.14.18.42.45_veh-12_03445_03902.pkl', 'rb'))
scale_factors = []
time_reference = 0
global_positions = []
for index in range(0, len(data)):
    global_positions.append(data[index]['ego2global'][:3, 3])

# Convert global_positions to numpy array for efficient vectorized operations
global_positions_np = np.array(global_positions)
closest_time_step = 0
for time_step in range(0, len(data)):

    current_ego2global = data[time_step]['ego2global'] #shape (4, 4)
    current_position = current_ego2global[:3, 3]  # First 3 elements of last column
    
    # Calculate distances to all positions using vectorized operations
    distances = np.linalg.norm(global_positions_np - current_position, axis=1)
    
    # Create mask for positions at least 15 meters away
    valid_distances_mask = distances >= 15
    
    # Find the closest valid position
    if np.any(valid_distances_mask):
        # Get indices of valid positions
        valid_indices = np.where(valid_distances_mask)[0]
        valid_distances = distances[valid_distances_mask]
        
        # Find the index of minimum distance among valid positions
        min_idx_in_valid = np.argmin(valid_distances)
        closest_time_step = valid_indices[min_idx_in_valid]
        displacement = valid_distances[min_idx_in_valid]
        
        historical_position = global_positions_np[closest_time_step]
    else:
        # No position found at least 15 meters away, skip this time step
        continue


    if displacement > 0:
        time_reference = closest_time_step
        current_f0_image_path = base_path + data[time_step]['cams']['CAM_F0']['data_path']
        historical_f0_image_path = base_path + data[time_reference]['cams']['CAM_F0']['data_path']
        image_names = [current_f0_image_path, historical_f0_image_path]

        images = load_and_preprocess_images(image_names).to(device)

        model = model.cuda()

        # Send images to GPU
        images = images.cuda()

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                # Predict attributes including cameras, depth maps, and point maps.
                predictions = model(images)




        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                images = images[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = model.aggregator(images)
                        
            # Predict Cameras
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            
            # Extract camera positions from extrinsic matrices (last column, first 3 elements)
            camera1_position = extrinsic[0, 0, :3, 3]  # First camera position
            camera2_position = extrinsic[0, 1, :3, 3]  # Second camera position
            
            # Calculate Euclidean distance between camera positions
            camera_displacement = torch.norm(camera1_position - camera2_position).item()
            # Predict Depth Maps
            #depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

            # Predict Point Maps
            #point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
                
            camera_displacement_list.append(camera_displacement)
            displacement_list.append(displacement)
            time_step_list.append(time_step)
            historical_image_path_list.append(historical_f0_image_path)
            scale_factor = displacement/camera_displacement
            scale_factors.append(scale_factor)
            print(f"time_step: {time_step}, scale_factor: {scale_factor:.3f}, real displacement: {displacement:.3f}, camera_displacement: {camera_displacement:.3f}")

            # save displacement and camera_displacement as a point and plot all points together plt.plot 

            # Construct 3D Points from Depth Maps and Cameras
            # which usually leads to more accurate 3D points than point map branch

            #point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0))

            # Predict Tracks
            # choose your own points to track, with shape (N, 2) for one scene
            #query_points = torch.FloatTensor([[100.0, 200.0], [60.72, 259.94]]).to(device)
            #track_list, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points[None])

# Save data to CSV
data_df = pd.DataFrame({
    'time_step': time_step_list,
    'historical_f0_image_path': historical_image_path_list,
    'real_displacement': displacement_list,
    'camera_displacement': camera_displacement_list,
    'scale_factor': scale_factors
})

csv_path = '/home/alienware3/navsim_workspace/displacement_analysis.csv'
data_df.to_csv(csv_path, index=False)
print(f"Data saved to {csv_path}")

# Create improved plots
plt.style.use('seaborn-v0_8')  # Use a modern style
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Camera Displacement Analysis', fontsize=16, fontweight='bold')

# Plot 1: Real vs Camera Displacement
axes[0, 0].scatter(displacement_list, camera_displacement_list, alpha=0.7, s=50, c='blue', edgecolors='black', linewidth=0.5)
axes[0, 0].set_xlabel('Real Displacement (m)', fontsize=12)
axes[0, 0].set_ylabel('Camera Displacement (m)', fontsize=12)
axes[0, 0].set_title('Real vs Camera Displacement', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Scale Factors vs Real Displacement
axes[0, 1].scatter(displacement_list, scale_factors, alpha=0.7, s=50, c='green', edgecolors='black', linewidth=0.5)
axes[0, 1].set_xlabel('Real Displacement (m)', fontsize=12)
axes[0, 1].set_ylabel('Scale Factor', fontsize=12)
axes[0, 1].set_title('Scale Factor vs Real Displacement', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Add horizontal line for average scale factor
avg_scale = np.mean(scale_factors)
axes[0, 1].axhline(y=avg_scale, color='red', linestyle='--', alpha=0.7, 
                   label=f'Average: {avg_scale:.3f}')
axes[0, 1].legend()

# Plot 3: Scale Factor Distribution
axes[1, 0].hist(scale_factors, bins=30, alpha=0.7, color='orange', edgecolor='black', linewidth=0.5)
axes[1, 0].set_xlabel('Scale Factor', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Scale Factor Distribution', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Add vertical line for average
axes[1, 0].axvline(x=avg_scale, color='red', linestyle='--', linewidth=2, 
                   label=f'Average: {avg_scale:.3f}')
axes[1, 0].legend()

# Plot 4: Time Series of Scale Factors
time_steps = list(range(len(scale_factors)))
axes[1, 1].plot(time_steps, scale_factors, 'o-', alpha=0.7, color='purple', 
                markersize=4, linewidth=1.5)
axes[1, 1].set_xlabel('Time Step', fontsize=12)
axes[1, 1].set_ylabel('Scale Factor', fontsize=12)
axes[1, 1].set_title('Scale Factor Over Time', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

# Add horizontal line for average
axes[1, 1].axhline(y=avg_scale, color='red', linestyle='--', alpha=0.7, 
                   label=f'Average: {avg_scale:.3f}')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('/home/alienware3/navsim_workspace/displacement_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n=== Summary Statistics ===")
print(f"Average scale factor: {avg_scale:.4f}")
print(f"Standard deviation: {np.std(scale_factors):.4f}")
print(f"Min scale factor: {np.min(scale_factors):.4f}")
print(f"Max scale factor: {np.max(scale_factors):.4f}")
print(f"Total samples: {len(scale_factors)}")

# Calculate correlation coefficient
correlation = np.corrcoef(displacement_list, camera_displacement_list)[0, 1]
print(f"Correlation between real and camera displacement: {correlation:.4f}")