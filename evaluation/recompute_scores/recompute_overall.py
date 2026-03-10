import pandas as pd
import numpy as np
import sys
import os

# Add current directory to sys.path to allow importing from evaluation.metrics
sys.path.append(os.getcwd())

from evaluation.metrics import compute_overall_score

def recompute_and_aggregate(file_paths):
    for file_path in file_paths:
        print(f"Processing {file_path}...")
        try:
            df = pd.read_csv(file_path)
            
            # Check if required columns exist
            required_columns = ['ade', 'fde', 'miss_rate', 'se', 'ac']
            if not all(col in df.columns for col in required_columns):
                print(f"Error: Missing columns in {file_path}. Required: {required_columns}")
                continue
            
            # Compute overall score for each row
            # The function signature is: compute_overall_score(ade, fde, miss_rate, se, ac, ...)
            # We use apply to run it row by row
            df['recomputed_overall_score'] = df.apply(
                lambda row: compute_overall_score(
                    row['ade'], 
                    row['fde'], 
                    row['miss_rate'], 
                    row['se'], 
                    row['ac']
                ), 
                axis=1
            )
            
            # Calculate average overall score
            average_score = df['recomputed_overall_score'].mean()
            
            print(f"Average Overall Score for {os.path.basename(file_path)}: {average_score}")
            print("-" * 30)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    files = [
        "./results/results_vipe.csv",
        "./results/results_vggt_20251113_152448.csv",
        "./results/results_spa_20251113_130742.csv"
    ]
    
    recompute_and_aggregate(files)
