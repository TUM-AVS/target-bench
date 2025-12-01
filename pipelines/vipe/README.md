# VIPE Trajectory Extraction

This directory contains code for extracting camera trajectories from video using VIPE's SLAM system.

## Overview

The VIPE (Visual Perception Engine) trajectory extraction system processes video files using a visual SLAM pipeline to estimate camera poses and extract trajectories with metric scale.

### Key Components

1. **VIPE SLAM System**: Visual SLAM pipeline based on DroidNet
2. **Trajectory Extraction**: Converts camera poses to trajectory coordinates
3. **Visualization**: Creates plots comparing predicted vs ground truth trajectories
4. **Scale Factor Computation**: Computes real-world metric scale if ground truth is available

## Directory Structure

```
pipelines/vipe/
├── extrinsic_path/
│   └── extract_extrinsic_path.py    # Main trajectory extraction module
├── configs/
│   └── vipe_trajectory_config.yml   # Configuration file
├── output/
│   └── trajectories/                 # Output directory for results
├── test_trajectory_extraction.py     # Test script
└── README.md                         # This file
```

## How It Works

### 1. VIPE SLAM Pipeline

The VIPE system processes video frames through:
- **Frontend**: Feature extraction and tracking (using DroidNet)
- **Backend**: Bundle adjustment to optimize camera poses
- **Output**: Camera poses (SE3 transformations) for each frame

### 2. Trajectory Extraction

The `extract_trajectory_from_poses()` function:
- Takes camera poses (rotation + translation matrices)
- Extracts camera positions in world coordinates
- Applies scale factor for real-world metrics
- Returns 3D trajectory and 2D projection for visualization

### 3. Scale Factor Computation

If ground truth metadata is available:
- Compares SLAM-estimated displacement vs actual displacement
- Computes: `scale_factor = real_displacement / camera_displacement`
- Applies scale to get metric trajectory

## Usage

### Basic Usage

```python
from extrinsic_path.extract_extrinsic_path import process_vipe_segments_for_trajectories

# Process all videos in a directory
trajectories, orientations, trajectories_2d, segment_info = process_vipe_segments_for_trajectories(
    segments_dir="/path/to/videos"
)
```

### Using Test Script

```bash
# Run interactive test menu
python test_trajectory_extraction.py

# Or run directly
python extrinsic_path/extract_extrinsic_path.py
```

### Command Line

```bash
# Process videos and extract trajectories
cd pipelines/vipe
python extrinsic_path/extract_extrinsic_path.py
```

## Input Data Format

### Video Files
- Supported formats: `.mp4`, `.MP4`
- Place videos in the configured `segments_dir`

### Optional Metadata (for scale factor computation)
- File: `{video_name}_metadata.json`
- Format:
```json
{
  "distance_first_to_current": {
    "distance_meters": 10.5
  }
}
```

### Optional Ground Truth Trajectory
- File: `{video_name}_trajectory.csv`
- Columns: `x`, `y` (in meters)

## Output Format

For each processed video, the following files are generated:

### 1. Trajectory Files
- `{segment}_trajectory.npy` - NumPy array of 3D positions
- `{segment}_orientations.npy` - NumPy array of rotation matrices
- `{segment}_trajectory.csv` - CSV with columns: x, y, z, frame_id
- `{segment}_trajectory.json` - JSON with full trajectory data and metadata

### 2. Visualization
- `{segment}_trajectory_plot.png` - 3-panel plot:
  - 3D trajectory view
  - 2D top-down view
  - Comparison with ground truth (if available)

### 3. Summary Report
- `summary_{timestamp}_trajectory_report.txt` - Statistics for all processed segments

## Configuration

Edit `configs/vipe_trajectory_config.yml` to customize:

- Input/output directories
- SLAM parameters (buffer size, resolution, etc.)
- Visualization settings
- Output formats

## Comparison with VGGT

| Feature | VGGT | VIPE |
|---------|------|------|
| Input | Video frames + metadata | Video files |
| Method | Geometry transformer model | Visual SLAM |
| Output | Predicted poses | Optimized poses |
| Scale | Computed from displacement | Computed from displacement |
| Tracking | Model-based | Feature-based |

## Example Workflow

```python
import sys
sys.path.append('models/vipe')

from extrinsic_path.extract_extrinsic_path import (
    process_vipe_segments_for_trajectories,
    save_trajectory_data,
    visualize_trajectory_with_gt,
    create_trajectory_summary_report
)

# Process videos
trajectories, orientations, trajectories_2d, segment_info = process_vipe_segments_for_trajectories(
    segments_dir="/path/to/videos"
)

# Save results
for i, (traj, orient, info) in enumerate(zip(trajectories, orientations, segment_info)):
    output_path = f"/path/to/output/{info['segment_name']}"
    
    # Save trajectory data
    save_trajectory_data(traj, orient, output_path, info)
    
    # Create visualization
    visualize_trajectory_with_gt(traj, None, output_path, info['segment_name'])

# Generate summary report
create_trajectory_summary_report(segment_info, "/path/to/output/summary")
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `buffer` size in config
- Lower `height` and `width` resolution
- Process videos one at a time

### No Poses Extracted
- Check video quality and motion
- Ensure sufficient visual features
- Try adjusting `init_disp` parameter

### Scale Factor is Infinity
- Metadata file not found or incorrect format
- Check that displacement values are valid
- Ensure SLAM produced valid poses

## Dependencies

Required packages:
- `torch` - PyTorch for deep learning
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib` - Visualization
- `omegaconf` - Configuration management
- VIPE package (in `models/vipe` directory)

## Notes

- The VIPE SLAM system is monocular by default, so absolute scale cannot be determined without ground truth
- Scale factor of 1.0 is used if no metadata is available
- Trajectories are in camera coordinate frame unless transformed
- For best results, ensure videos have good lighting and texture

## References

- VIPE Paper: [Link to paper if available]
- DroidNet: https://github.com/princeton-vl/DROID-SLAM
- Related: VGGT trajectory extraction in `../vggt/`
