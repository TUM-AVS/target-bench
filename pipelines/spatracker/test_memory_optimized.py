#!/usr/bin/env python3
"""
Memory-optimized test script for SpaTrackerV2 integration.
This script tests a single segment with the same approach as the working inference.py
"""

import os
import sys
import torch
sys.path.append('/home/hongyuan/world-decoder/SpaTrackerV2')

from scale_factor.compute_scale_factor import process_single_egowalk_segment

def main():
    # Test with a single segment using memory-optimized settings
    segment_path = "/home/hongyuan/world-decoder/dataset/EgoWalk_samples/2025_01_22__15_52_08_custom_segment_061_130"
    
    print(f"Testing memory-optimized single segment processing with SpaTrackerV2...")
    print(f"Segment path: {segment_path}")
    
    if not os.path.exists(segment_path):
        print(f"Error: Segment path does not exist: {segment_path}")
        return
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory before: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    try:
        result = process_single_egowalk_segment(segment_path)
        
        if result:
            print("\n" + "="*50)
            print("RESULTS (SpaTrackerV2 - Memory Optimized):")
            print("="*50)
            print(f"Segment: {result['segment_name']}")
            print(f"Real displacement: {result['real_displacement']:.4f} meters")
            print(f"Camera displacement: {result['camera_displacement']:.4f}")
            print(f"Scale factor: {result['scale_factor']:.4f}")
            print(f"Extrinsics shape: {result['extrinsic'].shape}")
            print(f"Intrinsics shape: {result['intrinsic'].shape}")
            print(f"Depths shape: {result['depths'].shape}")
            print("="*50)
            
            if torch.cuda.is_available():
                print(f"GPU memory after: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        else:
            print("Failed to process segment")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clear GPU memory after test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

