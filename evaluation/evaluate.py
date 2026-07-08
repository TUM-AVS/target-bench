#!/usr/bin/env python3
"""Unified Target-Bench evaluator for supported trajectory backbones."""

import argparse
import json
import os
import sys
import traceback

import pandas as pd
from tqdm.auto import tqdm


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVALUATION_DIR = os.path.join(PROJECT_ROOT, "evaluation")
for path in (PROJECT_ROOT, EVALUATION_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluation.backbones import BACKBONES, canonical_backbone_name, create_backbone
from evaluation.common import (
    compute_summary_statistics,
    compute_trajectory_metrics,
    create_evaluation_visualizations,
    find_world_model_videos,
    get_segment_dirs,
    load_evaluation_config,
    load_ground_truth_trajectory,
    load_ground_truth_trajectory_current_to_end,
    save_results,
)


def _print_eval_config(eval_config, eval_config_path, gt_trajectory_scope):
    weights = eval_config["overall_score"]["weights"]
    print(f"\nLoaded evaluation config from: {eval_config_path}")
    print(f"  Miss threshold: {eval_config['metrics']['miss_threshold']}m")
    print(
        "  Overall score weights: "
        f"ADE={weights['ade']}, FDE={weights['fde']}, MR={weights['mr']}, "
        f"SE={weights['se']}, AC={weights['ac']}"
    )
    print(f"  GT trajectory scope: {gt_trajectory_scope}")


def _required_segment_paths(segment_dir, segment_name):
    return {
        "first_frame": os.path.join(segment_dir, f"{segment_name}_first_frame.png"),
        "current_frame": os.path.join(segment_dir, f"{segment_name}_current_frame.png"),
        "metadata": os.path.join(segment_dir, f"{segment_name}_metadata.json"),
    }


def _load_gt(segment_dir, segment_name, metadata, gt_trajectory_scope):
    if gt_trajectory_scope == "full":
        return load_ground_truth_trajectory(segment_dir, segment_name)
    if gt_trajectory_scope == "current_to_end":
        return load_ground_truth_trajectory_current_to_end(segment_dir, segment_name, metadata)
    raise ValueError(f"Unknown gt_trajectory_scope: {gt_trajectory_scope}")


def evaluate_world_models(
    backbone,
    segments_dir=None,
    wm_videos_dir=None,
    output_dir=None,
    config_path=None,
    eval_config_path=None,
    num_samples=None,
    model_id=None,
    checkpoint_path=None,
    gt_trajectory_scope="current_to_end",
    model_names=None,
    make_plots=True,
):
    """Evaluate world-model videos with one supported trajectory backbone."""
    backbone = canonical_backbone_name(backbone)
    runner = create_backbone(
        backbone,
        project_root=PROJECT_ROOT,
        config_path=config_path,
        model_id=model_id,
        checkpoint_path=checkpoint_path,
    )

    segments_dir = segments_dir or os.path.join(PROJECT_ROOT, "dataset/Benchmark")
    wm_videos_dir = wm_videos_dir or os.path.join(PROJECT_ROOT, "dataset/wm_videos")
    output_dir = output_dir or os.path.join(PROJECT_ROOT, "evaluation_results", runner.output_subdir)
    eval_config_path = eval_config_path or runner.eval_config_path

    print("=" * 80)
    print(f"World Models Evaluation using {runner.display_name}")
    print("=" * 80)

    eval_config = load_evaluation_config(eval_config_path)
    _print_eval_config(eval_config, eval_config_path, gt_trajectory_scope)

    print("\n" + "=" * 80)
    print(f"Loading {runner.display_name} Model")
    print("=" * 80)
    runner.setup()

    os.makedirs(output_dir, exist_ok=True)
    segment_dirs = get_segment_dirs(segments_dir, num_samples)
    wm_videos = find_world_model_videos(wm_videos_dir)
    if model_names:
        requested = set(model_names)
        missing = sorted(requested - set(wm_videos))
        if missing:
            print(f"Warning: requested models not found: {missing}")
        wm_videos = {name: videos for name, videos in wm_videos.items() if name in requested}
    print(f"\nFound {len(wm_videos)} world models: {list(wm_videos.keys())}")

    all_results = {
        "backbone": backbone,
        "backbone_display_name": runner.display_name,
        "by_model": {},
        "by_segment": {},
        "summary": {},
        "gt_trajectory_scope": gt_trajectory_scope,
    }

    for wm_model_name, model_videos in wm_videos.items():
        print(f"\n{'=' * 80}")
        print(f"Evaluating Model: {wm_model_name}")
        print(f"{'=' * 80}")

        all_results["by_model"][wm_model_name] = {}

        for segment_dir in tqdm(segment_dirs, desc=f"Processing {wm_model_name}"):
            segment_name = os.path.basename(segment_dir.rstrip("/"))
            if segment_name not in model_videos:
                print(f"Warning: No video found for segment {segment_name} in model {wm_model_name}")
                continue

            video_path = model_videos[segment_name]
            print(f"\nProcessing: {segment_name}")
            print(f"  WM Video: {os.path.basename(video_path)}")

            paths = _required_segment_paths(segment_dir, segment_name)
            if not all(os.path.exists(path) for path in paths.values()):
                print("  Missing required segment files")
                continue
            if not os.path.exists(video_path):
                print(f"  Video path does not exist: {video_path}")
                continue

            try:
                with open(paths["metadata"], "r") as f:
                    metadata = json.load(f)

                pred = runner.run(segment_dir, segment_name, video_path, metadata, output_dir)
                gt_trajectory_2d = _load_gt(segment_dir, segment_name, metadata, gt_trajectory_scope)
                if gt_trajectory_2d is None:
                    print("  No ground truth trajectory found")
                    continue

                pred_trajectory_2d = pred["trajectory_2d"]
                metrics = compute_trajectory_metrics(pred_trajectory_2d, gt_trajectory_2d, eval_config)

            except Exception as exc:
                print(f"  Error processing: {exc}")
                traceback.print_exc()
                continue

            result_entry = {
                "segment_name": segment_name,
                "model_name": wm_model_name,
                "scale_factor": float(pred["scale_factor"]),
                "real_displacement": float(pred["real_displacement"]),
                "camera_displacement": float(pred["camera_displacement"]),
                "trajectory_length": float(pred["trajectory_length"]),
                "gt_trajectory_scope": gt_trajectory_scope,
                "metrics": metrics,
                "trajectory": pred["trajectory"].tolist(),
                "trajectory_2d": pred_trajectory_2d.tolist(),
                "gt_trajectory_2d": gt_trajectory_2d.tolist(),
            }

            all_results["by_model"][wm_model_name][segment_name] = result_entry
            all_results["by_segment"].setdefault(segment_name, {})[wm_model_name] = result_entry

            print(
                f"  Overall: {metrics['overall_score']:.4f} | "
                f"ADE: {metrics['ade']:.4f}m, FDE: {metrics['fde']:.4f}m, "
                f"MR: {metrics['miss_rate']:.2f}%, SE: {metrics['se']:.4f}, AC: {metrics['ac']:.4f}"
            )

    compute_summary_statistics(all_results)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    plot_fn = None
    if make_plots:
        plot_fn = lambda results, out_dir, ts: create_evaluation_visualizations(
            results,
            out_dir,
            ts,
            title_prefix=runner.display_name,
        )
    save_results(all_results, output_dir, timestamp, plot_fn)
    return all_results


def build_parser(default_backbone=None):
    parser = argparse.ArgumentParser(description="Evaluate Target-Bench world-model videos")
    default_backbone = canonical_backbone_name(default_backbone or "vggt_omega")
    parser.add_argument(
        "--backbone",
        choices=sorted(BACKBONES.keys()),
        default=default_backbone,
        help=f"Trajectory backbone to use, default: {default_backbone}",
    )
    parser.add_argument("-n", "--num_samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--segments_dir", type=str, default=None, help="Directory containing benchmark segments")
    parser.add_argument("--wm_videos_dir", type=str, default=None, help="Directory containing world-model videos")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results")
    parser.add_argument("--config_path", type=str, default=None, help="Backbone pipeline config path")
    parser.add_argument("--eval_config_path", type=str, default=None, help="Evaluation metrics config path")
    parser.add_argument("--model_id", type=str, default=None, help="DA3 model ID override")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Local VGGT/VGGT-Omega checkpoint override")
    parser.add_argument("--models", nargs="+", default=None, help="Evaluate only selected world-model directory names")
    parser.add_argument(
        "--gt_trajectory_scope",
        choices=["full", "current_to_end"],
        default="current_to_end",
        help="Ground-truth trajectory scope for metrics",
    )
    parser.add_argument("--no_plots", action="store_true", help="Skip PNG visualization generation")
    return parser


def main(default_backbone=None):
    parser = build_parser(default_backbone=default_backbone)
    args = parser.parse_args()

    results = evaluate_world_models(
        backbone=args.backbone,
        segments_dir=args.segments_dir,
        wm_videos_dir=args.wm_videos_dir,
        output_dir=args.output_dir,
        config_path=args.config_path,
        eval_config_path=args.eval_config_path,
        num_samples=args.num_samples,
        model_id=args.model_id,
        checkpoint_path=args.checkpoint_path,
        gt_trajectory_scope=args.gt_trajectory_scope,
        model_names=args.models,
        make_plots=not args.no_plots,
    )

    output_dir = args.output_dir
    if output_dir is None:
        runner = create_backbone(canonical_backbone_name(args.backbone), PROJECT_ROOT)
        output_dir = os.path.join(PROJECT_ROOT, "evaluation_results", runner.output_subdir)

    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print(f"Backbone: {results['backbone_display_name']}")
    print(f"Results saved to: {output_dir}")
    print("\nTo view results:")
    print(f"  - Summary CSV: {output_dir}/model_summary_*.csv")
    print(f"  - Detailed results: {output_dir}/segment_model_results_*.csv")
    print(f"  - Full JSON: {output_dir}/evaluation_results_*.json")
    print(f"  - Visualizations: {output_dir}/*.png")


if __name__ == "__main__":
    main()
