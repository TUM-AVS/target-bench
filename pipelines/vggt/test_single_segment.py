#!/usr/bin/env python3
"""
Test script to process a single EgoWalk segment
"""

import os
import sys
sys.path.append('/home/hongyuan/world-decoder/vggt_code')

from scale_factor.compute_scale_factor import process_single_egowalk_segment

def main():
    # Test with a single segment
    segment_path = "/home/hongyuan/world-decoder/dataset/EgoWalk_samples/2025_01_17__13_15_23_custom_segment_040_088"
    
    print(f"Testing single segment processing...")
    print(f"Segment path: {segment_path}")
    
    if not os.path.exists(segment_path):
        print(f"Error: Segment path does not exist: {segment_path}")
        return
    
    try:
        result = process_single_egowalk_segment(segment_path)
        
        if result:
            print("\n" + "="*50)
            print("RESULTS:")
            print("="*50)
            print(f"Segment: {result['segment_name']}")
            print(f"Real displacement: {result['real_displacement']:.4f} meters")
            print(f"Camera displacement: {result['camera_displacement']:.4f}")
            print(f"Scale factor: {result['scale_factor']:.4f}")
            print("="*50)
        else:
            print("Failed to process segment")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
