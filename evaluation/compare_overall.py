#!/usr/bin/env python3
"""
Compare Overall Scores across different tracking methods (VGGT, SPA, ViPE)

This script visualizes the overall scores from three different tracking methods
in a consistent format with three subplots.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Define project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_summary_data(csv_path):
    """Load model summary data from CSV."""
    df = pd.read_csv(csv_path)
    return df


def find_latest_model_summary(base_dir, method_subdir, file_pattern):
    """
    Find the latest model_summary CSV file in a directory based on timestamp.
    
    Args:
        base_dir: Base evaluation results directory
        method_subdir: Subdirectory for the method (e.g., 'vggt', 'spa', 'vipe')
        file_pattern: Pattern to match files (e.g., 'model_summary_*.csv')
    
    Returns:
        Path to the latest CSV file, or None if not found
    """
    search_dir = os.path.join(base_dir, method_subdir)
    pattern = os.path.join(search_dir, file_pattern)
    
    # Find all matching files
    files = glob.glob(pattern)
    
    if not files:
        print(f"Warning: No files found matching {pattern}")
        return None
    
    # Sort files by timestamp (extracted from filename)
    # Files are typically named: model_summary_YYYYMMDD_HHMMSS.csv
    # or model_summary_spa_YYYYMMDD_HHMMSS.csv
    files_sorted = sorted(files, reverse=True)
    
    latest_file = files_sorted[0]
    print(f"Found latest {method_subdir} file: {os.path.basename(latest_file)}")
    
    return latest_file


def create_overall_comparison_plot(vggt_csv, spa_csv, vipe_csv, output_dir=None):
    """
    Create comparison plot of overall scores across VGGT, SPA, and ViPE.
    
    Args:
        vggt_csv: Path to VGGT model summary CSV
        spa_csv: Path to SPA model summary CSV
        vipe_csv: Path to ViPE model summary CSV
        output_dir: Directory to save output plot (optional)
    """
    # Load data
    df_vggt = load_summary_data(vggt_csv)
    df_spa = load_summary_data(spa_csv)
    df_vipe = load_summary_data(vipe_csv)
    
    # Get all unique models across all three methods
    all_models = set(df_vggt['model'].tolist() + df_spa['model'].tolist() + df_vipe['model'].tolist())
    
    # Sort models: gt_video first, then others alphabetically
    models_sorted = sorted(all_models, key=lambda x: (x != 'gt_video', x))
    
    # Create consistent color mapping for all models
    color_palette = cm.tab20(np.linspace(0, 1, 20))
    model_colors = {}
    for i, model_name in enumerate(models_sorted):
        model_colors[model_name] = color_palette[i % 20]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(26,8))
    
    # Determine y-axis limits based on VGGT data
    vggt_max_score = df_vggt['overall_score_mean'].max()
    y_max = vggt_max_score * 1.1  # Add 10% padding
    
    # Plot data for each tracking method
    datasets = [
        (df_vggt, 'VGGT Overall Score', axes[0]),
        (df_spa, 'SpaTracker Overall Score', axes[1]),
        (df_vipe, 'ViPE Overall Score', axes[2])
    ]
    i = 0
    for df, title, ax in datasets:
        # Filter models present in this dataset
        valid_models = [m for m in models_sorted if m in df['model'].values]
        
        # Get overall scores
        means = []
        for model in valid_models:
            score = df[df['model'] == model]['overall_score_mean'].values[0]
            means.append(score)
        
        if means:
            x_pos = np.arange(len(valid_models))
            # Get colors for each model
            colors = [model_colors[m] for m in valid_models]
            
            # Plot bar chart
            ax.bar(x_pos, means, alpha=0.7, color=colors)
            ax.set_xticks(x_pos + 0.4)  # Shift ticks slightly right
            ax.set_xticklabels(valid_models, rotation=75, ha='right', fontsize=30)
            if i == 0:
                ax.set_ylabel('Overall Score', fontsize=30)
            else:
                ax.set_ylabel('', fontsize=30)
            ax.set_title(title, fontweight='bold', fontsize=36)
            ax.tick_params(axis='y', labelsize=26)
            ax.grid(True, alpha=0.3)
            
            # Set consistent y-axis limits across all plots based on VGGT
            ax.set_ylim(0, y_max)
        i += 1
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15) # Add horizontal space between subplots
    
    # Save plot
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'overall_score_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved overall score comparison plot to: {output_path}")
    
    # plt.show()


def main():
    """Main function."""
    # Base directory for evaluation results
    base_dir = os.path.join(PROJECT_ROOT, 'evaluation_results')
    
    # Automatically find the latest model summary files for each method
    # Ensure directories exist before searching
    if not os.path.exists(base_dir):
        print(f"Base directory not found: {base_dir}")
        return

    vggt_csv = find_latest_model_summary(base_dir, 'vggt', 'model_summary_*.csv')
    spa_csv = find_latest_model_summary(base_dir, 'spa', 'model_summary_spa_*.csv')
    vipe_csv = find_latest_model_summary(base_dir, 'vipe', 'model_summary_*.csv')
    
    # Check if all files were found
    if not all([vggt_csv, spa_csv, vipe_csv]):
        print("Error: Could not find all required model summary files")
        print(f"VGGT: {vggt_csv}")
        print(f"SPA: {spa_csv}")
        print(f"ViPE: {vipe_csv}")
        return
    
    # Create comparison plot
    create_overall_comparison_plot(vggt_csv, spa_csv, vipe_csv)


if __name__ == "__main__":
    main()