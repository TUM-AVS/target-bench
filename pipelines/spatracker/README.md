# World Decoder - SpaTrackerV2 Integration

This directory contains the SpaTrackerV2-based implementation for world decoding, mirroring the structure and functionality of `pipelines/vggt` but using SpaTrackerV2 models instead of VGGT.

## Directory Structure

```
pipelines/spatracker/
├── configs/
│   └── spatracker_pcd_configs.yml     # Configuration for SpaTrackerV2 processing
├── scale_factor/
│   └── compute_scale_factor.py        # Scale factor computation using SpaTrackerV2
├── extrinsic_path/
│   └── extract_extrinsic_path.py      # Trajectory extraction from SpaTrackerV2 poses
├── output/                            # Output directory for results
├── test_single_segment.py             # Test script for single segment processing
├── test_trajectory_extraction.py      # Test script for trajectory extraction
└── world-dec.py                       # Placeholder for additional functionality
```

## Key Differences from VGGT Version

### Models Used
- **VGGT Version**: Uses `VGGT` model from `models/vggt/`
- **SpaTrackerV2 Version**: Uses `VGGT4Track` and `Predictor` models from `models/spatracker/`

### Core Functionality
1. **Pose Estimation**: Uses SpaTrackerV2's VGGT4Track model for camera pose prediction
2. **Depth Estimation**: Leverages SpaTrackerV2's depth prediction capabilities
3. **Tracking Integration**: Incorporates tracking features for better temporal consistency
4. **Scale Factor Computation**: Maintains same methodology but with SpaTrackerV2 outputs

### Configuration
The configuration file `spatracker_pcd_configs.yml` includes:
- SpaTrackerV2-specific model settings
- Track mode selection (offline/online)
- Video processing parameters
- Point cloud post-processing options

## Usage

### Process Single Segment
```bash
python test_single_segment.py
```

### Extract Trajectories
```bash
python test_trajectory_extraction.py
```

### Process All Segments
```bash
cd scale_factor/
python compute_scale_factor.py
```

Or:
```bash
cd extrinsic_path/
python extract_extrinsic_path.py
```

## Dependencies

- SpaTrackerV2 models and utilities
- PyTorch with CUDA support
- Standard computer vision libraries (cv2, numpy, etc.)
- Visualization libraries (matplotlib, optional)

## Model Integration

### SpaTrackerV2 Models
1. **VGGT4Track**: Primary model for pose and depth estimation
   - Model: `"Yuxihenry/SpatialTrackerV2_Front"`
   - Provides camera poses, intrinsics, depth maps, and confidence

2. **Predictor**: Tracking model for enhanced temporal consistency
   - Offline: `"Yuxihenry/SpatialTrackerV2-Offline"`
   - Online: `"Yuxihenry/SpatialTrackerV2-Online"`

### Data Flow
1. Video frames → VGGT4Track → Poses, Depths, Intrinsics
2. Scale factor computation using pose predictions
3. Trajectory extraction with metric scaling
4. Visualization and analysis

## Output Formats

Same as VGGT version:
- JSON files with scale factor results
- CSV files with trajectory data
- Visualization plots comparing predicted vs ground truth
- Summary reports with statistics

## Compatibility

This implementation maintains full compatibility with the EgoWalk dataset structure and produces outputs in the same format as the VGGT version, enabling direct comparison between the two approaches.

