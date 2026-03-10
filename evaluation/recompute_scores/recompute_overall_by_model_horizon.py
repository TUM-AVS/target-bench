import pandas as pd
import numpy as np
import sys
import os

# Add current directory to sys.path to allow importing from evaluation.metrics
sys.path.append(os.getcwd())

from evaluation.metrics import compute_overall_score

def recompute_and_aggregate_by_model_horizon(file_paths):
    for file_path in file_paths:
        print(f"Processing {os.path.basename(file_path)}...")
        try:
            df = pd.read_csv(file_path)
            
            # Check if required columns exist
            required_columns = ['ade', 'fde', 'miss_rate', 'se', 'ac', 'model']
            if not all(col in df.columns for col in required_columns):
                print(f"Error: Missing columns in {file_path}. Required: {required_columns}")
                continue
            
            # Compute overall score for each row
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
            
            # Group by model and calculate average score
            model_scores = df.groupby('model')['recomputed_overall_score'].mean().reset_index()
            # Sort by score descending
            model_scores = model_scores.sort_values(by='recomputed_overall_score', ascending=False)
            
            print(f"Average Overall Score by Model for {os.path.basename(file_path)}:")
            print(model_scores.to_string(index=False))
            print("-" * 50)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    files = [
        "./results/results_horizon_4s_20251114_020803.csv",
        "./results/results_horizon_2s_20251113_221816.csv"
    ]
    
    recompute_and_aggregate_by_model_horizon(files)
