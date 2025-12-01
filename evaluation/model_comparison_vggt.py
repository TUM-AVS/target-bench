#!/usr/bin/env python3
"""
Generate model comparison diagram from evaluation results CSV.

This script reads model summary CSV and creates a comparison bar chart
with wan2.2-ti2v-5B-F-D highlighted at the last column.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse


def create_model_comparison(csv_path, output_path=None):
    """
    Create model comparison bar chart from summary CSV.
    
    Args:
        csv_path: Path to model_summary CSV file
        output_path: Path to save output PNG (optional, auto-generated if None)
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Extract model names
    models = df['model'].tolist()
    
    # Sort models: gt_video first, then others alphabetically
    models_sorted = sorted(models, key=lambda x: (x != 'gt_video', x))
    
    # Move wan2.2-ti2v-5B-F-D to last position
    target_model = 'wan2.2-ti2v-5B-F-D'
    if target_model in models_sorted:
        models_sorted.remove(target_model)
        models_sorted.append(target_model)
    
    models = models_sorted
    
    # Create consistent color mapping for all models
    color_palette = cm.tab20(np.linspace(0, 1, 20))
    model_colors = {}
    for i, model_name in enumerate(models):
        model_colors[model_name] = color_palette[i % 20]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(22, 16))
    
    metrics_to_plot = ['overall_score', 'ade', 'fde', 'miss_rate', 'se', 'ac']
    titles = ['Overall Score', 'ADE (m)', 'FDE (m)', 'Miss Rate (%)', 'SE', 'AC']
    
    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[idx // 3, idx % 3]
        
        # Get data for each model in the specified order
        means = []
        valid_models = []
        for model in models:
            model_data = df[df['model'] == model]
            if len(model_data) > 0:
                mean_val = model_data[f'{metric}_mean'].values[0]
                if pd.notna(mean_val):
                    means.append(mean_val)
                    valid_models.append(model)
        
        if means:
            x_pos = np.arange(len(valid_models))
            colors = [model_colors[m] for m in valid_models]
            
            # Plot bars
            ax.bar(x_pos, means, alpha=0.7, color=colors)
            ax.set_xticks(x_pos + 0.4)
            
            # Set x-tick labels with bold only for target model
            labels = []
            for model in valid_models:
                if model == target_model:
                    labels.append(model)
                else:
                    labels.append(model)
            
            # Set labels with custom formatting
            ax.set_xticklabels(labels, rotation=75, ha='right', fontsize=30)
            
            # Bold only the target model label
            for i, (label, model) in enumerate(zip(ax.get_xticklabels(), valid_models)):
                if model == target_model:
                    label.set_weight('bold')
            
            ax.set_title(f'{title}', 
                        fontweight='bold' if metric == 'overall_score' else 'normal',
                        fontsize=36)
            ax.tick_params(axis='y', labelsize=26)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15)
    
    # Generate output path if not provided
    if output_path is None:
        csv_dir = os.path.dirname(csv_path)
        csv_basename = os.path.basename(csv_path)
        timestamp = csv_basename.replace('model_summary_', '').replace('.csv', '')
        output_path = os.path.join(csv_dir, f"model_comparison_{timestamp}.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved model comparison plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate model comparison diagram from evaluation results CSV'
    )
    parser.add_argument(
        'csv_path',
        type=str,
        nargs='?',
        default='/home/hongyuan/world-decoder/evaluation_results/vggt/model_summary_20251113_152448.csv',
        help='Path to model_summary CSV file (default: evaluation_results/vggt/model_summary_20251113_152448.csv)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output path for PNG file (optional, auto-generated if not provided)'
    )
    
    args = parser.parse_args()
    
    create_model_comparison(args.csv_path, args.output)


if __name__ == "__main__":
    main()

