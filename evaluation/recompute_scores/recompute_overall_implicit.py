import pandas as pd
import numpy as np
import sys
import os

# Add current directory to sys.path to allow importing from evaluation.metrics
sys.path.append(os.getcwd())

from evaluation.metrics import compute_overall_score

def recompute_and_aggregate_implicit(file_path):
    print(f"Processing {os.path.basename(file_path)}...")
    try:
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
        required_columns = ['ade', 'fde', 'miss_rate', 'se', 'ac', 'model', 'is_explicit']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: Missing columns in {file_path}. Required: {required_columns}")
            return
        
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
        
        # Get unique values of is_explicit
        is_explicit_values = df['is_explicit'].unique()
        is_explicit_values.sort()
        
        for is_explicit_val in is_explicit_values:
            group_name = "Explicit" if is_explicit_val == 1 else "Implicit" if is_explicit_val == 0 else f"is_explicit={is_explicit_val}"
            print(f"\n=== Results for {group_name} (is_explicit={is_explicit_val}) ===")
            
            # Filter by is_explicit
            df_group = df[df['is_explicit'] == is_explicit_val]
            
            # Group by model and calculate average score
            model_scores = df_group.groupby('model')['recomputed_overall_score'].mean().reset_index()
            # Sort by score descending
            model_scores = model_scores.sort_values(by='recomputed_overall_score', ascending=False)
            
            print(model_scores.to_string(index=False))
            print("-" * 50)
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    file_path = "./results/results_implicit_20251113_155527.csv"
    recompute_and_aggregate_implicit(file_path)
