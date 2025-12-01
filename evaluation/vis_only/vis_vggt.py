#!/usr/bin/env python3
"""
Standalone visualization script for VGGT evaluation results.

This script loads evaluation results from the vggt results folder and generates visualizations.
"""

import os
import sys
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
from pathlib import Path


def find_latest_results(results_dir):
    """Find the latest evaluation results JSON file."""
    json_files = glob.glob(os.path.join(results_dir, "evaluation_results_*.json"))
    if not json_files:
        print(f"No evaluation results found in {results_dir}")
        return None
    
    # Sort by modification time, return the latest
    latest_file = max(json_files, key=os.path.getmtime)
    print(f"Loading results from: {latest_file}")
    return latest_file


def load_results(json_path):
    """Load evaluation results from JSON file."""
    with open(json_path, 'r') as f:
        results = json.load(f)
    return results


def create_model_comparison(all_results, output_dir, output_suffix=""):
    """Create model comparison bar charts."""
    # Sort models: gt_video first, then others alphabetically
    models = list(all_results['summary'].keys())
    models_sorted = sorted(models, key=lambda x: (x != 'gt_video', x))
    
    # Create consistent color mapping for all models
    # Use tab20 for more distinct colors
    color_palette = cm.tab20(np.linspace(0, 1, 20))
    model_colors = {}
    for i, model_name in enumerate(models_sorted):
        model_colors[model_name] = color_palette[i % 20]
    
    # 1. Model comparison - bar charts (using metrics from metrics.md + overall score)
    fig, axes = plt.subplots(2, 3, figsize=(22, 16))
    
    metrics_to_plot = ['overall_score', 'ade', 'fde', 'miss_rate', 'se', 'ac']
    titles = ['Overall Score', 'ADE (m)', 'FDE (m)', 'Miss Rate (%)', 'SE', 'AC']
    
    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[idx // 3, idx % 3]
        
        # Filter valid models and sort (gt_video first)
        valid_models = [m for m in models_sorted if all_results['summary'][m][metric]['mean'] is not None]
        means = [all_results['summary'][m][metric]['mean'] for m in valid_models]
        
        if means:
            x_pos = np.arange(len(valid_models))
            # Get colors for each model
            colors = [model_colors[m] for m in valid_models]
            
            # Plot without error bars
            ax.bar(x_pos, means, alpha=0.7, color=colors)
            ax.set_xticks(x_pos + 0.4)  # Shift ticks slightly right
            ax.set_xticklabels(valid_models, rotation=75, ha='right', fontsize=30)  # 2x bigger
            # 2x bigger ax.set_ylabel(title, fontsize=30)  
            ax.set_title(f'{title}', 
                        fontweight='bold' if metric == 'overall_score' else 'normal',
                        fontsize=36)  # 2x bigger
            ax.tick_params(axis='y', labelsize=26)  # 2x bigger for y-axis ticks
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15) # Add horizontal space between subplots
    output_path = os.path.join(output_dir, f"model_comparison{output_suffix}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved model comparison plot: {output_path}")
    plt.close()
    
    return model_colors


def create_trajectory_plots(all_results, model_colors, output_dir, output_suffix=""):
    """Create trajectory comparison plots for each segment."""
    for segment_name, segment_results in all_results['by_segment'].items():
        fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
        plt.subplots_adjust(left=0.02, right=0.85)
        
        # Plot GT trajectory
        first_result = list(segment_results.values())[0]
        gt_traj = np.array(first_result['gt_trajectory_2d'])
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], 'k--', linewidth=3, label='Ground Truth', zorder=10)
        ax.scatter(gt_traj[0, 0], gt_traj[0, 1], c='green', s=200, marker='o', edgecolors='black', linewidth=2, label='GT Start', zorder=11)
        ax.scatter(gt_traj[-1, 0], gt_traj[-1, 1], c='red', s=200, marker='s', edgecolors='black', linewidth=2, label='GT End', zorder=11)
        
        # Collect all x values to adjust xlim and reduce left padding
        all_x_values = gt_traj[:, 0].tolist()
        
        # Plot predicted trajectories for each model (using consistent colors)
        for model_name, result in segment_results.items():
            pred_traj = np.array(result['trajectory_2d'])
            color = model_colors[model_name]  # Use consistent color from model_colors
            
            ax.plot(pred_traj[:, 0], pred_traj[:, 1], '-', color=color, linewidth=2, 
                   alpha=0.7, label=f'{model_name}')
            ax.scatter(pred_traj[0, 0], pred_traj[0, 1], c=color, s=80, marker='o', zorder=5)
            ax.scatter(pred_traj[-1, 0], pred_traj[-1, 1], c=color, s=80, marker='s', zorder=5)
            
            # Collect x values
            all_x_values.extend(pred_traj[:, 0].tolist())
        
        # Set xlim to reduce left empty space
        min_x = min(all_x_values)
        max_x = max(all_x_values)
        x_range = max_x - min_x
        ax.set_xlim(min_x - 0.01 * x_range, max_x + 0.05 * x_range)  # Minimal left margin
        
        ax.set_xlabel('X (meters)', fontsize=30)  # 2x bigger
        ax.set_ylabel('Y (meters)', fontsize=30)  # 2x bigger
        ax.set_title(
            f'Trajectory Comparison\nVGGT - {segment_name}',
            fontsize=36,
            pad=20,
            loc='center',
            x=0.5
        )  # 2x bigger, two rows, more padding
        ax.tick_params(axis='both', labelsize=26)  # 2x bigger for tick labels
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24)  # Legend outside on right
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"trajectory_{segment_name}{output_suffix}.png")
        pos = ax.get_position()
        ax.set_position([0.02, pos.y0, 0.70, pos.height])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Saved {len(all_results['by_segment'])} trajectory comparison plots")


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualizations for VGGT evaluation results'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default="/home/hongyuan/world-decoder/evaluation_results/vggt",
        help='Directory containing evaluation results'
    )
    parser.add_argument(
        '--output_suffix',
        type=str,
        default="_viz",
        help='Suffix to add to output filenames'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("VGGT Evaluation Results Visualization")
    print("="*80)
    
    # Find and load latest results
    results_json = find_latest_results(args.results_dir)
    if results_json is None:
        return
    
    all_results = load_results(results_json)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    model_colors = create_model_comparison(all_results, args.results_dir, args.output_suffix)
    create_trajectory_plots(all_results, model_colors, args.results_dir, args.output_suffix)
    
    print("\n" + "="*80)
    print("Visualization Complete!")
    print("="*80)
    print(f"Visualizations saved to: {args.results_dir}")


if __name__ == "__main__":
    main()

