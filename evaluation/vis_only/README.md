# Visualization Scripts for Evaluation Results

This directory contains standalone visualization scripts that load evaluation results from result folders and generate visualizations.

## Overview

All visualization scripts are now synchronized with the visualization code from `target_eval_vggt.py`, ensuring consistent:
- Color schemes across all models
- Font sizes (30 for labels, 36 for titles, 26 for tick labels, 24 for legends)
- Layout and formatting
- Bar chart styles with proper spacing

## Individual Visualization Scripts

Each script loads the latest JSON results from its corresponding result folder and generates:
1. Model comparison bar charts (6 metrics: Overall Score, ADE, FDE, Miss Rate, SE, AC)
2. Trajectory comparison plots for each segment

### Available Scripts

1. **vis_vggt.py** - VGGT evaluation results
2. **vis_vipe.py** - VIPE evaluation results
3. **vis_spa.py** - SpaTracker evaluation results
4. **vis_vggt_implicit.py** - VGGT Implicit evaluation results (includes explicit vs implicit comparison)
5. **vis_vggt_horizon_2s.py** - VGGT Horizon 2s evaluation results
6. **vis_vggt_horizon_4s.py** - VGGT Horizon 4s evaluation results

## Usage

### Running Individual Scripts

Each script can be run independently:

```bash
# Generate visualizations for VGGT results
python vis_vggt.py

# Generate visualizations for VIPE results with custom suffix
python vis_vipe.py --output_suffix "_new"

# Generate visualizations for SpaTracker results from custom directory
python vis_spa.py --results_dir /path/to/spa/results
```

### Command Line Arguments

All scripts support the following arguments:

- `--results_dir`: Directory containing evaluation results (default: specific to each script)
- `--output_suffix`: Suffix to add to output filenames (default: "_viz")

### Running All Visualizations at Once

Use the master script to generate visualizations for all result folders:

```bash
# Generate all visualizations
python vis_all.py

# Generate all visualizations with custom base directory
python vis_all.py --base_dir /path/to/evaluation_results

# Generate all visualizations with custom suffix
python vis_all.py --output_suffix "_new"
```

## Output Files

Each script generates the following files in its corresponding result folder:

### Standard Scripts (VGGT, VIPE, SpaTracker, Horizon 2s, Horizon 4s)
- `model_comparison{suffix}.png` - Bar charts comparing all models across 6 metrics
- `trajectory_{segment_name}{suffix}.png` - Trajectory comparison for each segment

### VGGT Implicit Script
- `model_comparison_overall{suffix}.png` - Overall comparison across all samples
- `model_comparison_explicit_vs_implicit{suffix}.png` - Side-by-side comparison of explicit vs implicit
- `trajectory_{segment_name}{suffix}.png` - Trajectory comparison for each segment

## Features

### Consistent Visualization Style

All scripts use the same visualization style from `target_eval_vggt.py`:
- **Figure size**: 22x16 inches for model comparison, 12x10 for trajectories
- **Font sizes**: 
  - Axis labels: 30
  - Titles: 36
  - Tick labels: 26
  - Legends: 24
- **Colors**: tab20 colormap with consistent color assignment per model
- **Layout**: 2x3 subplot grid for metrics, proper spacing with wspace=0.15
- **Rotation**: 75 degrees for x-axis labels

### Model Ordering

Models are sorted with:
1. `gt_video` first (if present)
2. All other models alphabetically

### Color Consistency

Colors are assigned consistently across all plots:
- Same model always gets the same color
- Uses matplotlib's tab20 colormap for distinct colors

## Requirements

Required Python packages:
- numpy
- matplotlib
- json (standard library)
- glob (standard library)
- argparse (standard library)

## Directory Structure

Expected directory structure:
```
evaluation_results/
├── vggt/
│   ├── evaluation_results_*.json
│   └── ... (generated visualizations)
├── vipe/
│   ├── evaluation_results_*.json
│   └── ... (generated visualizations)
├── spa/
│   ├── evaluation_results_spa_*.json
│   └── ... (generated visualizations)
├── vggt_implicit/
│   ├── evaluation_results_*.json
│   └── ... (generated visualizations)
├── vggt_horizon_2s/
│   ├── evaluation_results_horizon_2s_*.json
│   └── ... (generated visualizations)
└── vggt_horizon_4s/
    ├── evaluation_results_horizon_4s_*.json
    └── ... (generated visualizations)
```

## Notes

- Scripts automatically find the latest JSON results file in each directory
- If no results are found, the script will print a warning and exit
- All visualizations are saved to the same directory as the input JSON file
- Output files use high DPI (300) for publication-quality figures




