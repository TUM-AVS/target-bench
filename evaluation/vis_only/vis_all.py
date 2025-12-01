#!/usr/bin/env python3
"""
Master visualization script to generate visualizations for all evaluation results.

This script runs all visualization scripts for each result folder.
"""

import os
import sys
import subprocess
import argparse


def run_visualization(script_name, results_dir, output_suffix=""):
    """Run a visualization script."""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    cmd = [sys.executable, script_path, '--results_dir', results_dir]
    if output_suffix:
        cmd.extend(['--output_suffix', output_suffix])
    
    print(f"\nRunning {script_name}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {script_name} failed with error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualizations for all evaluation results'
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default="/home/hongyuan/world-decoder/evaluation_results",
        help='Base directory containing all result folders'
    )
    parser.add_argument(
        '--output_suffix',
        type=str,
        default="_viz",
        help='Suffix to add to output filenames'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Master Visualization Script - Generating All Visualizations")
    print("="*80)
    
    # Define all result folders and their corresponding visualization scripts
    visualizations = [
        ('vis_vggt.py', os.path.join(args.base_dir, 'vggt')),
        ('vis_vipe.py', os.path.join(args.base_dir, 'vipe')),
        ('vis_spa.py', os.path.join(args.base_dir, 'spa')),
        ('vis_vggt_implicit.py', os.path.join(args.base_dir, 'vggt_implicit')),
        ('vis_vggt_horizon_2s.py', os.path.join(args.base_dir, 'vggt_horizon_2s')),
        ('vis_vggt_horizon_4s.py', os.path.join(args.base_dir, 'vggt_horizon_4s')),
    ]
    
    # Track results
    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    # Run each visualization
    for script_name, results_dir in visualizations:
        if not os.path.exists(results_dir):
            print(f"\n⚠ Skipping {script_name}: Directory not found: {results_dir}")
            skipped_count += 1
            continue
        
        success = run_visualization(script_name, results_dir, args.output_suffix)
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # Print summary
    print("\n" + "="*80)
    print("Master Visualization Complete!")
    print("="*80)
    print(f"Successfully generated: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Skipped: {skipped_count}")
    print(f"\nAll visualizations saved to their respective result folders in: {args.base_dir}")


if __name__ == "__main__":
    main()




