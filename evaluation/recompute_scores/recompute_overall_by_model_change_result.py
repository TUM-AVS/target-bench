import pandas as pd
import numpy as np
import sys
import os
import glob

# Add current directory to sys.path to allow importing from evaluation.metrics
sys.path.append(os.getcwd())

from evaluation.metrics import compute_overall_score

def recompute_and_update_summary(tasks):
    """
    tasks: list of tuples (input_result_csv, summary_dir)
    """
    for input_csv, summary_dir in tasks:
        print(f"Processing input: {os.path.basename(input_csv)}")
        
        # 1. Recompute scores from input result file
        try:
            df = pd.read_csv(input_csv)
            
            # Check if required columns exist
            required_columns = ['ade', 'fde', 'miss_rate', 'se', 'ac', 'model']
            if not all(col in df.columns for col in required_columns):
                print(f"Error: Missing columns in {input_csv}. Required: {required_columns}")
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
            # Rename column for easier merging/updating
            model_scores = model_scores.rename(columns={'recomputed_overall_score': 'new_overall_score_mean'})
            
            print("Recomputed scores:")
            print(model_scores)
            
        except Exception as e:
            print(f"Error processing input {input_csv}: {e}")
            continue

        # 2. Find and update summary file
        try:
            # Find csv files in summary directory
            summary_files = glob.glob(os.path.join(summary_dir, "*.csv"))
            if not summary_files:
                print(f"No summary CSV found in {summary_dir}")
                continue
            
            # Assuming there is only one summary file or we take the first one
            # Ideally we should match names but the naming convention differs slightly
            summary_csv_path = summary_files[0]
            print(f"Updating summary file: {summary_csv_path}")
            
            summary_df = pd.read_csv(summary_csv_path)
            
            if 'model' not in summary_df.columns:
                print(f"Error: 'model' column missing in summary file {summary_csv_path}")
                continue
            
            # Update overall_score_mean
            # We can merge and then update, or iterate
            
            # Create a dictionary for mapping
            score_map = dict(zip(model_scores['model'], model_scores['new_overall_score_mean']))
            
            original_scores = summary_df.set_index('model')['overall_score_mean'].to_dict() if 'overall_score_mean' in summary_df.columns else {}
            
            # Update the column
            # If 'overall_score_mean' doesn't exist, create it
            if 'overall_score_mean' not in summary_df.columns:
                summary_df['overall_score_mean'] = np.nan
            
            def update_score(row):
                model = row['model']
                if model in score_map:
                    return score_map[model]
                return row['overall_score_mean']

            summary_df['overall_score_mean'] = summary_df.apply(update_score, axis=1)
            
            # Verify changes
            for model, new_score in score_map.items():
                if model in original_scores:
                    old_score = original_scores[model]
                    print(f"  Model {model}: {old_score:.6f} -> {new_score:.6f}")
                else:
                    print(f"  Model {model}: (new) -> {new_score:.6f}")

            # Save back to CSV
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"Successfully updated {summary_csv_path}")
            print("-" * 50)

        except Exception as e:
            print(f"Error updating summary for {summary_dir}: {e}")


if __name__ == "__main__":
    base_dir = "."
    
    tasks = [
        (
            os.path.join(base_dir, "results/results_vipe.csv"),
            os.path.join(base_dir, "evaluation_results/vipe")
        ),
        (
            os.path.join(base_dir, "results/results_vggt_20251113_152448.csv"),
            os.path.join(base_dir, "evaluation_results/vggt")
        ),
        (
            os.path.join(base_dir, "results/results_spa_20251113_130742.csv"),
            os.path.join(base_dir, "evaluation_results/spa")
        )
    ]
    
    recompute_and_update_summary(tasks)
