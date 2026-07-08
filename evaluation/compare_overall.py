#!/usr/bin/env python3
"""
Compare Overall Scores across supported backbones (VGGT, DA3, VGGT-Omega)

This script visualizes the overall scores from three supported backbones
in a consistent format with three subplots.
"""

import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Define project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_summary_data(csv_path):
    """Load model summary data from CSV."""
    df = pd.read_csv(csv_path)
    if "category" in df.columns:
        df = df[df["category"].fillna("overall") == "overall"].copy()
    if "model" in df.columns:
        df["model"] = df["model"].replace({"gt_videos": "gt_video"})
    return df


def find_latest_model_summary(base_dir, method_subdir, file_pattern):
    """
    Find the latest model_summary CSV file in a directory based on timestamp.
    
    Args:
        base_dir: Base evaluation results directory
        method_subdir: Subdirectory for the method (e.g., 'vggt', 'da3', 'vggt_omega')
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


def create_overall_comparison_plot(vggt_csv, da3_csv, vggt_omega_csv, output_path=None, models=None):
    """
    Create comparison plot of overall scores across VGGT, DA3, and VGGT-Omega.
    
    Args:
        vggt_csv: Path to VGGT model summary CSV
        da3_csv: Path to DA3 model summary CSV
        vggt_omega_csv: Path to VGGT-Omega model summary CSV
        output_path: Path to save output plot (optional)
        models: Optional ordered model list. Models not in any CSV are skipped.
    """
    # Load data
    df_vggt = load_summary_data(vggt_csv)
    df_da3 = load_summary_data(da3_csv)
    df_vggt_omega = load_summary_data(vggt_omega_csv)
    
    if models is None:
        # Get all unique models across all three methods
        all_models = set(df_vggt['model'].tolist() + df_da3['model'].tolist() + df_vggt_omega['model'].tolist())
        models_sorted = sorted(all_models, key=lambda x: (x != 'gt_video', x))
    else:
        models = ["gt_video" if m == "gt_videos" else m for m in models]
        available_models = set(df_vggt['model'].tolist() + df_da3['model'].tolist() + df_vggt_omega['model'].tolist())
        models_sorted = [m for m in models if m in available_models]
    
    # Create consistent color mapping for all models
    color_palette = cm.tab20(np.linspace(0, 1, 20))
    model_colors = {}
    for i, model_name in enumerate(models_sorted):
        model_colors[model_name] = color_palette[i % 20]
    
    # Keep this shorter than the old single-row plot while leaving enough room for bars.
    fig, axes = plt.subplots(1, 3, figsize=(25, 5.6))
    
    # Plot data for each tracking method
    datasets = [
        (df_vggt, 'VGGT Overall Score', axes[0]),
        (df_da3, 'DA3 Overall Score', axes[1]),
        (df_vggt_omega, 'VGGT-Omega Overall Score', axes[2])
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
            ax.set_xticks([])
            if i == 0:
                ax.set_ylabel('Overall Score', fontsize=30)
            else:
                ax.set_ylabel('', fontsize=30)
            ax.set_title(title, fontweight='bold', fontsize=36)
            ax.tick_params(axis='y', labelsize=26)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        i += 1

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=model_colors[m], alpha=0.7)
        for m in models_sorted
    ]
    fig.legend(
        legend_handles,
        models_sorted,
        loc="lower center",
        ncol=6,
        fontsize=29,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
        columnspacing=1.2,
        handlelength=1.0,
        handletextpad=0.4,
    )

    plt.tight_layout(rect=[0, 0.26, 1, 1])
    plt.subplots_adjust(wspace=0.15)
    
    # Save plot
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, 'evaluation_results', 'overall_score_comparison.png')
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved overall score comparison plot to: {output_path}")
    
    # plt.show()


def create_vggt_model_compare_plot(csv_path, output_path, models=None):
    """
    Create the single VGGT model comparison plot used by the paper figure.

    Args:
        csv_path: Path to VGGT model_summary CSV.
        output_path: Path to save the PNG.
        models: Optional ordered model list. Models not in the CSV are skipped.
    """
    df = load_summary_data(csv_path)

    if models is None:
        models = sorted(df["model"].tolist(), key=lambda x: (x not in {"gt_video", "gt_videos"}, x))
    else:
        models = ["gt_video" if m == "gt_videos" else m for m in models]
        models = [m for m in models if m in set(df["model"])]

    # Keep the paper legend text stable while using the current result artifact name.
    display_names = {
        "gt_videos": "gt_video",
    }

    target_model = "wan2.2-ti2v-5B-F-D"
    if target_model in models:
        models = [m for m in models if m != target_model] + [target_model]

    color_palette = cm.tab20(np.linspace(0, 1, 20))
    model_colors = {model_name: color_palette[i % 20] for i, model_name in enumerate(models)}

    fig, axes = plt.subplots(2, 3, figsize=(25, 9.7))

    metrics_to_plot = ["overall_score", "ade", "fde", "miss_rate", "se", "ac"]
    titles = ["Overall Score", "ADE (m)", "FDE (m)", "Miss Rate (%)", "SE", "AC"]

    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[idx // 3, idx % 3]

        valid_models = []
        means = []
        for model in models:
            model_data = df[df["model"] == model]
            if len(model_data) == 0:
                continue
            mean_val = model_data[f"{metric}_mean"].values[0]
            if pd.notna(mean_val):
                valid_models.append(model)
                means.append(mean_val)

        if not means:
            continue

        x_pos = np.arange(len(valid_models))
        colors = [model_colors[m] for m in valid_models]
        ax.bar(x_pos, means, alpha=0.7, color=colors)
        ax.set_xticks([])

        ax.set_title(
            title,
            fontweight="bold" if metric == "overall_score" else "normal",
            fontsize=36,
        )
        ax.tick_params(axis="y", labelsize=26)
        ax.grid(True, alpha=0.3)
        if metric in {"overall_score", "se", "ac"}:
            ax.set_ylim(0, 1)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=model_colors[m], alpha=0.7)
        for m in models
    ]
    legend_labels = [display_names.get(m, m) for m in models]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=6,
        fontsize=29,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        columnspacing=1.2,
        handlelength=1.0,
        handletextpad=0.4,
    )

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.subplots_adjust(wspace=0.15, hspace=0.35)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved supported-backbone comparison plot to: {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["overall", "vggt-model"],
        default="overall",
        help="Plot mode. 'overall' compares VGGT, DA3, and VGGT-Omega.",
    )
    parser.add_argument("--vggt-csv", default=None, help="VGGT model summary CSV.")
    parser.add_argument("--da3-csv", default=None, help="DA3 model summary CSV for --mode overall.")
    parser.add_argument("--vggt-omega-csv", default=None, help="VGGT-Omega model summary CSV for --mode overall.")
    parser.add_argument("--base-dir", default=None, help="Base evaluation_results directory for auto-discovery.")
    parser.add_argument("--output", default=None, help="Output PNG path.")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated ordered model list.",
    )
    args = parser.parse_args()

    if args.mode == "vggt-model":
        if args.vggt_csv is None:
            raise SystemExit("--vggt-csv is required for --mode vggt-model")
        if args.output is None:
            raise SystemExit("--output is required for --mode vggt-model")
        models = [m.strip() for m in args.models.split(",")] if args.models else None
        create_vggt_model_compare_plot(args.vggt_csv, args.output, models=models)
        return

    # Base directory for evaluation results
    base_dir = args.base_dir or os.path.join(PROJECT_ROOT, 'evaluation_results')
    
    # Automatically find the latest model summary files for each method
    # Ensure directories exist before searching
    if not os.path.exists(base_dir):
        print(f"Base directory not found: {base_dir}")
        return

    vggt_csv = args.vggt_csv or find_latest_model_summary(base_dir, 'vggt', 'model_summary_*.csv')
    da3_csv = args.da3_csv or find_latest_model_summary(base_dir, 'da3', 'model_summary_*.csv')
    vggt_omega_csv = args.vggt_omega_csv or find_latest_model_summary(base_dir, 'vggt_omega', 'model_summary_*.csv')
    
    # Check if all files were found
    if not all([vggt_csv, da3_csv, vggt_omega_csv]):
        print("Error: Could not find all required model summary files")
        print(f"VGGT: {vggt_csv}")
        print(f"DA3: {da3_csv}")
        print(f"VGGT-Omega: {vggt_omega_csv}")
        return
    
    # Create comparison plot
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    create_overall_comparison_plot(vggt_csv, da3_csv, vggt_omega_csv, output_path=args.output, models=models)


if __name__ == "__main__":
    main()
