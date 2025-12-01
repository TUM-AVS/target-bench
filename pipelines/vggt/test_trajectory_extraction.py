#!/usr/bin/env python3
"""
Test script for trajectory extraction from EgoWalk segments
"""

import os
import sys
sys.path.append('/home/hongyuan/world-decoder/vggt_code')

from extrinsic_path.extract_extrinsic_path import process_single_segment_demo, main

def test_single_segment():
    """Test trajectory extraction on a single segment."""
    print("="*60)
    print("Testing Single Segment Trajectory Extraction")
    print("="*60)
    
    # Test with the sample segment
    segment_name = "2025_01_17__13_15_23_custom_segment_040_088"
    process_single_segment_demo(segment_name)

def test_all_segments():
    """Test trajectory extraction on all segments."""
    print("="*60)
    print("Testing All Segments Trajectory Extraction")
    print("="*60)
    
    # Run the main function which processes all segments
    main()

def main_menu():
    """Simple menu for testing."""
    print("EgoWalk Trajectory Extraction Test Menu")
    print("="*40)
    print("1. Test single segment")
    print("2. Test all segments")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            try:
                test_single_segment()
            except Exception as e:
                print(f"Error in single segment test: {e}")
                import traceback
                traceback.print_exc()
        elif choice == '2':
            try:
                test_all_segments()
            except Exception as e:
                print(f"Error in all segments test: {e}")
                import traceback
                traceback.print_exc()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    print("EgoWalk Trajectory Extraction Test")
    print("This script tests the trajectory extraction functionality")
    print("using the process_egowalk_segments function from compute_scale_factor.py")
    print()
    
    # Check if dataset exists
    segments_dir = "/home/hongyuan/world-decoder/dataset/Target_samples"
    if not os.path.exists(segments_dir):
        print(f"Error: EgoWalk samples directory not found: {segments_dir}")
        sys.exit(1)
    
    # Check if sample segment exists
    sample_segment = os.path.join(segments_dir, "2025_01_17__13_15_23_custom_segment_040_088")
    if not os.path.exists(sample_segment):
        print(f"Warning: Sample segment not found: {sample_segment}")
    
    main_menu()
