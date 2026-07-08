"""Shared helpers for target-bench trajectory evaluation."""

import glob
import json
import os

import numpy as np
import pandas as pd
import yaml

try:
    from .metrics import compute_all_metrics
except ImportError:
    from metrics import compute_all_metrics


def load_evaluation_config(config_path):
    """Load a YAML metric configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_world_model_videos(wm_videos_dir):
    """Find generated videos organized as {model_name}/{segment_name}.mp4."""
    wm_videos = {}
    if not os.path.exists(wm_videos_dir):
        print(f"Warning: World model videos directory not found: {wm_videos_dir}")
        return wm_videos

    def should_skip_model_dir(model_name):
        lowered = model_name.lower()
        if model_name.startswith("."):
            return True
        return any(marker in lowered for marker in ("backup", "rebuilt", "wrong"))

    for model_dir in sorted(glob.glob(os.path.join(wm_videos_dir, "*/"))):
        if not os.path.isdir(model_dir):
            continue

        model_name = os.path.basename(model_dir.rstrip("/"))
        if should_skip_model_dir(model_name):
            print(f"Skipping non-current video directory: {model_name}")
            continue

        video_files = sorted(glob.glob(os.path.join(model_dir, "*.mp4")))
        if not video_files:
            print(f"Skipping video directory with no mp4 files: {model_name}")
            continue

        wm_videos[model_name] = {}
        for video_path in video_files:
            segment_name = os.path.splitext(os.path.basename(video_path))[0]
            wm_videos[model_name][segment_name] = video_path

        print(f"Found {len(wm_videos[model_name])} videos for model: {model_name}")

    return wm_videos


def get_segment_dirs(segments_dir, num_samples=None):
    """Return sorted segment directories, optionally truncated for smoke tests."""
    segment_dirs = sorted([d for d in glob.glob(os.path.join(segments_dir, "*/")) if os.path.isdir(d)])
    print(f"\nFound {len(segment_dirs)} segments in total")

    if num_samples is not None and num_samples > 0:
        segment_dirs = segment_dirs[:num_samples]
        print(f"Limiting evaluation to {len(segment_dirs)} samples (--num_samples={num_samples})")
    else:
        print(f"Evaluating all {len(segment_dirs)} segments")

    return segment_dirs


def load_ground_truth_trajectory_current_to_end(segment_dir, segment_name, metadata):
    """Load the GT trajectory slice from the current frame through the segment end."""
    trajectory_path = os.path.join(segment_dir, f"{segment_name}_trajectory.csv")
    if not os.path.exists(trajectory_path):
        print(f"Warning: Ground truth trajectory file not found: {trajectory_path}")
        return None

    try:
        gt_df = pd.read_csv(trajectory_path)
        if "x" not in gt_df.columns or "y" not in gt_df.columns:
            print(f"Warning: GT trajectory CSV missing x/y columns: {trajectory_path}")
            return None

        start_abs = int(metadata.get("video_start_idx", metadata.get("trajectory_start_frame", 0)))
        current_abs = int(metadata["distance_current_to_end"]["current_frame"]["video_index"])
        end_abs = int(metadata["distance_current_to_end"]["end_frame"]["video_index"])
        start_idx = current_abs - start_abs
        end_idx = end_abs - start_abs

        if start_idx < 0 or end_idx < start_idx or start_idx >= len(gt_df):
            print(
                f"Warning: Invalid current-to-end GT slice for {segment_name}: "
                f"start_idx={start_idx}, end_idx={end_idx}, len={len(gt_df)}"
            )
            return None

        end_idx = min(end_idx, len(gt_df) - 1)
        gt_trajectory_2d = gt_df.iloc[start_idx:end_idx + 1][["x", "y"]].values
        gt_trajectory_2d = gt_trajectory_2d[~np.isnan(gt_trajectory_2d).any(axis=1)]

        if len(gt_trajectory_2d) == 0:
            print(f"Warning: Empty current-to-end GT trajectory for {segment_name}")
            return None

        return gt_trajectory_2d - gt_trajectory_2d[0]

    except Exception as exc:
        print(f"Error loading current-to-end GT trajectory from {trajectory_path}: {exc}")
        return None


def load_ground_truth_trajectory(segment_dir, segment_name):
    """Load the full GT trajectory for a benchmark segment."""
    trajectory_path = os.path.join(segment_dir, f"{segment_name}_trajectory.csv")
    if not os.path.exists(trajectory_path):
        print(f"Warning: Ground truth trajectory file not found: {trajectory_path}")
        return None

    try:
        gt_df = pd.read_csv(trajectory_path)
        if "x" not in gt_df.columns or "y" not in gt_df.columns:
            print(f"Warning: GT trajectory CSV missing x/y columns: {trajectory_path}")
            return None

        gt_trajectory_2d = gt_df[["x", "y"]].values
        gt_trajectory_2d = gt_trajectory_2d[~np.isnan(gt_trajectory_2d).any(axis=1)]
        if len(gt_trajectory_2d) == 0:
            print(f"Warning: Empty GT trajectory for {segment_name}")
            return None

        return gt_trajectory_2d

    except Exception as exc:
        print(f"Error loading GT trajectory from {trajectory_path}: {exc}")
        return None


def compute_trajectory_metrics(pred_trajectory_2d, gt_trajectory_2d, eval_config):
    """Compute target-bench metrics from a YAML evaluation config."""
    if len(pred_trajectory_2d) == 0 or len(gt_trajectory_2d) == 0:
        return {
            "ade": float("inf"),
            "fde": float("inf"),
            "miss_rate": 100.0,
            "se": 0.0,
            "ac": 0.0,
            "overall_score": 0.0,
            "valid": False,
            "pred_points": len(pred_trajectory_2d),
            "gt_points": len(gt_trajectory_2d),
        }

    metrics_config = eval_config["metrics"]
    overall_config = eval_config["overall_score"]
    metrics = compute_all_metrics(
        pred_trajectory_2d,
        gt_trajectory_2d,
        miss_threshold=metrics_config["miss_threshold"],
        interpolate_mode=metrics_config["interpolate_mode"],
        se_sigma=metrics_config["se_sigma"],
        ac_params=metrics_config["ac"],
        overall_score_params={
            "tau_ade": overall_config["tau_ade"],
            "tau_fde": overall_config["tau_fde"],
            "weights": overall_config["weights"],
        },
    )
    metrics["valid"] = True
    return metrics


def compute_summary_statistics(all_results):
    """Populate per-model summary statistics in-place."""
    print(f"\n{'=' * 80}")
    print("Computing Summary Statistics")
    print(f"{'=' * 80}")

    for model_name, model_results in all_results["by_model"].items():
        if not model_results:
            continue

        buckets = {key: [] for key in ["overall_score", "ade", "fde", "miss_rate", "se", "ac"]}
        for segment_result in model_results.values():
            metrics = segment_result["metrics"]
            if metrics["valid"]:
                for key in buckets:
                    buckets[key].append(metrics[key])

        summary = {"num_segments": len(model_results), "num_valid": len(buckets["ade"])}
        for key, values in buckets.items():
            summary[key] = {
                "mean": float(np.mean(values)) if values else None,
                "median": float(np.median(values)) if values else None,
                "std": float(np.std(values)) if values else None,
                "min": float(np.min(values)) if values else None,
                "max": float(np.max(values)) if values else None,
            }

        all_results["summary"][model_name] = summary
        print(f"\n{model_name}:")
        print(f"  Segments processed: {summary['num_valid']}/{summary['num_segments']}")
        if summary["overall_score"]["mean"] is not None:
            print(f"  Overall Score: {summary['overall_score']['mean']:.4f} +/- {summary['overall_score']['std']:.4f}")
            print(f"  ADE: {summary['ade']['mean']:.4f} +/- {summary['ade']['std']:.4f} m")
            print(f"  FDE: {summary['fde']['mean']:.4f} +/- {summary['fde']['std']:.4f} m")
            print(f"  Miss Rate: {summary['miss_rate']['mean']:.2f} +/- {summary['miss_rate']['std']:.2f} %")
            print(f"  SE: {summary['se']['mean']:.4f} +/- {summary['se']['std']:.4f}")
            print(f"  AC: {summary['ac']['mean']:.4f} +/- {summary['ac']['std']:.4f}")


def _convert_for_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def save_results(all_results, output_dir, timestamp, plot_fn=None):
    """Save detailed JSON and summary CSV files, then optionally make plots."""
    results_json_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
    with open(results_json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_convert_for_json)
    print(f"\nSaved detailed results to: {results_json_path}")

    create_summary_csv(all_results, output_dir, timestamp)

    if plot_fn is not None:
        try:
            plot_fn(all_results, output_dir, timestamp)
        except Exception as exc:
            print(f"Warning: Could not create visualizations: {exc}")


def create_summary_csv(all_results, output_dir, timestamp):
    """Create per-segment and per-model summary CSV files."""
    segment_model_data = []
    for segment_name, segment_results in all_results["by_segment"].items():
        for model_name, result in segment_results.items():
            metrics = result["metrics"]
            segment_model_data.append(
                {
                    "segment": segment_name,
                    "model": model_name,
                    "scale_factor": result.get("scale_factor"),
                    "real_displacement": result.get("real_displacement"),
                    "camera_displacement": result.get("camera_displacement"),
                    "trajectory_length": result.get("trajectory_length"),
                    "overall_score": metrics["overall_score"],
                    "ade": metrics["ade"],
                    "fde": metrics["fde"],
                    "miss_rate": metrics["miss_rate"],
                    "se": metrics["se"],
                    "ac": metrics["ac"],
                    "pred_points": metrics["pred_points"],
                    "gt_points": metrics["gt_points"],
                }
            )

    segment_csv_path = os.path.join(output_dir, f"segment_model_results_{timestamp}.csv")
    pd.DataFrame(segment_model_data).to_csv(segment_csv_path, index=False)
    print(f"Saved per-segment results to: {segment_csv_path}")

    summary_data = []
    for model_name, summary in all_results["summary"].items():
        summary_data.append(
            {
                "model": model_name,
                "num_segments": summary["num_segments"],
                "num_valid": summary["num_valid"],
                "overall_score_mean": summary["overall_score"]["mean"],
                "overall_score_std": summary["overall_score"]["std"],
                "ade_mean": summary["ade"]["mean"],
                "ade_std": summary["ade"]["std"],
                "fde_mean": summary["fde"]["mean"],
                "fde_std": summary["fde"]["std"],
                "miss_rate_mean": summary["miss_rate"]["mean"],
                "miss_rate_std": summary["miss_rate"]["std"],
                "se_mean": summary["se"]["mean"],
                "se_std": summary["se"]["std"],
                "ac_mean": summary["ac"]["mean"],
                "ac_std": summary["ac"]["std"],
            }
        )

    summary_csv_path = os.path.join(output_dir, f"model_summary_{timestamp}.csv")
    pd.DataFrame(summary_data).to_csv(summary_csv_path, index=False)
    print(f"Saved model summary to: {summary_csv_path}")


def create_evaluation_visualizations(all_results, output_dir, timestamp, title_prefix="Trajectory"):
    """Create metric bar charts and per-segment trajectory plots."""
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    models = sorted(all_results["summary"].keys(), key=lambda x: (x != "gt_videos", x))
    if not models:
        return

    color_palette = cm.tab20(np.linspace(0, 1, 20))
    model_colors = {model_name: color_palette[i % 20] for i, model_name in enumerate(models)}

    metrics_to_plot = ["overall_score", "ade", "fde", "miss_rate", "se", "ac"]
    titles = ["Overall Score", "ADE (m)", "FDE (m)", "Miss Rate (%)", "SE", "AC"]
    fig, axes = plt.subplots(2, 3, figsize=(22, 16))

    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[idx // 3, idx % 3]
        valid_models = [m for m in models if all_results["summary"][m][metric]["mean"] is not None]
        means = [all_results["summary"][m][metric]["mean"] for m in valid_models]
        if means:
            x_pos = np.arange(len(valid_models))
            ax.bar(x_pos, means, alpha=0.7, color=[model_colors[m] for m in valid_models])
            ax.set_xticks(x_pos)
            ax.set_xticklabels(valid_models, rotation=75, ha="right", fontsize=18)
            ax.set_title(title, fontweight="bold" if metric == "overall_score" else "normal", fontsize=22)
            ax.tick_params(axis="y", labelsize=16)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"model_comparison_{timestamp}.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved model comparison plot")

    for segment_name, segment_results in all_results["by_segment"].items():
        fig, ax = plt.subplots(figsize=(12, 10))

        first_result = next(iter(segment_results.values()))
        gt_traj = np.array(first_result["gt_trajectory_2d"])
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], "k--", linewidth=3, label="Ground Truth", zorder=10)
        ax.scatter(gt_traj[0, 0], gt_traj[0, 1], c="green", s=120, marker="o", label="GT Start", zorder=11)
        ax.scatter(gt_traj[-1, 0], gt_traj[-1, 1], c="red", s=120, marker="s", label="GT End", zorder=11)

        all_x_values = gt_traj[:, 0].tolist()
        for model_name, result in segment_results.items():
            pred_traj = np.array(result["trajectory_2d"])
            color = model_colors[model_name]
            ax.plot(pred_traj[:, 0], pred_traj[:, 1], "-", color=color, linewidth=2, alpha=0.7, label=model_name)
            ax.scatter(pred_traj[0, 0], pred_traj[0, 1], c=[color], s=70, marker="o", zorder=5)
            ax.scatter(pred_traj[-1, 0], pred_traj[-1, 1], c=[color], s=70, marker="s", zorder=5)
            all_x_values.extend(pred_traj[:, 0].tolist())

        min_x = min(all_x_values)
        max_x = max(all_x_values)
        x_range = max(max_x - min_x, 1e-6)
        ax.set_xlim(min_x - 0.01 * x_range, max_x + 0.05 * x_range)

        ax.set_xlabel("X (meters)", fontsize=22)
        ax.set_ylabel("Y (meters)", fontsize=22)
        ax.set_title(f"Trajectory Comparison\n{title_prefix} - {segment_name}", fontsize=26, pad=20)
        ax.tick_params(axis="both", labelsize=18)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=16)
        ax.axis("equal")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"trajectory_{segment_name}_{timestamp}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Saved {len(all_results['by_segment'])} trajectory comparison plots")
