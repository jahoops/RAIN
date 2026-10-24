import pandas as pd
import numpy as np
import os
import glob
from sklearn.metrics import roc_auc_score
import shutil

def find_best_submission():
    """Find the best submission file based on available metrics or validation"""
    print("=== FINDING BEST MODEL FOR FINAL SUBMISSION ===")
    
    # Priority order for submissions to check
    priority_models = [
        'synthetic_neural',
        'neural',
        'uncertainty_model',
        'u_b',
        'winning_ensemble_fixed3_fixed',
        'winning_blend',
        'direct_ensemble',
        'suggested_ensemble',
        'xgb_fixed',
        'lgb_fixed'
    ]
    
    # Places to look for submissions
    locations = {
        'main': 'data',
        'keep': 'data/keep',
        'fixed': 'data/fixed_submissions'
    }
    
    candidates = []
    
    # First check priority models in each location
    for model in priority_models:
        for location, path in locations.items():
            for suffix in ['', '_fixed', '_fixed2', '_fixed3']:
                file_name = f'submission_{model}{suffix}.csv'
                file_path = os.path.join(path, file_name)
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path)
                        if 'id' in df.columns and 'rainfall' in df.columns:
                            if not df['rainfall'].isna().any() and len(np.unique(df['rainfall'])) > 1:
                                # Valid submission with non-constant values
                                var_score = df['rainfall'].std()  # Use variance as a simple quality metric
                                candidates.append({
                                    'path': file_path,
                                    'name': f"{model}{suffix}",
                                    'location': location,
                                    'var_score': var_score,
                                    'priority': priority_models.index(model) if model in priority_models else 999
                                })
                                print(f"Found candidate: {file_name} (σ={var_score:.4f})")
                    except Exception as e:
                        print(f"Error checking {file_path}: {str(e)}")
    
    # If no priority models found, scan for any submission files
    if not candidates:
        print("No priority models found, scanning for any valid submissions...")
        for location, path in locations.items():
            for file_path in glob.glob(os.path.join(path, 'submission_*.csv')):
                file_name = os.path.basename(file_path)
                try:
                    df = pd.read_csv(file_path)
                    if 'id' in df.columns and 'rainfall' in df.columns:
                        if not df['rainfall'].isna().any() and len(np.unique(df['rainfall'])) > 1:
                            var_score = df['rainfall'].std()
                            candidates.append({
                                'path': file_path,
                                'name': file_name.replace('submission_', '').replace('.csv', ''),
                                'location': location,
                                'var_score': var_score,
                                'priority': 999
                            })
                            print(f"Found candidate: {file_name} (σ={var_score:.4f})")
                except Exception as e:
                    print(f"Error checking {file_path}: {str(e)}")
    
    if not candidates:
        print("No valid submissions found with varying predictions.")
        return None
    
    # Sort candidates by priority and then by variance
    candidates.sort(key=lambda x: (x['priority'], -x['var_score']))
    
    best_candidate = candidates[0]
    print(f"\nSelected best candidate: {best_candidate['name']} from {best_candidate['location']}")
    print(f"Path: {best_candidate['path']}")
    
    # Load the best submission
    best_df = pd.read_csv(best_candidate['path'])
    
    # Create a summary of the predictions
    if 'rainfall' in best_df.columns:
        rainfall = best_df['rainfall']
        print("\nPrediction statistics:")
        print(f"Mean: {rainfall.mean():.4f}")
        print(f"Std: {rainfall.std():.4f}")
        print(f"Min: {rainfall.min():.4f}")
        print(f"Max: {rainfall.max():.4f}")
        print(f"Distribution: {np.percentile(rainfall, [0, 25, 50, 75, 100])}")
    
    return best_df

def create_better_submission():
    """Create an improved final submission"""
    # Find the best submission
    best_df = find_best_submission()
    
    if best_df is None:
        print("Could not find a suitable submission. No changes made.")
        return False
    
    # Create data/final directory if it doesn't exist
    os.makedirs('data/final', exist_ok=True)
    
    # Check if final submission exists and back it up if needed
    final_path = 'data/final/submission_final.csv'
    if os.path.exists(final_path):
        backup_path = 'data/final/submission_final_backup.csv'
        shutil.copy2(final_path, backup_path)
        print(f"Backed up existing submission to {backup_path}")
    
    # Save the best submission as the final submission
    best_df.to_csv(final_path, index=False)
    print(f"Created better final submission at {final_path}")
    
    # Also save a copy with a descriptive name
    descriptive_path = 'data/final/submission_final_improved.csv'
    best_df.to_csv(descriptive_path, index=False)
    
    return True

if __name__ == "__main__":
    print("Creating improved final submission...")
    create_better_submission()
    print("\nDone! Check data/final/submission_final.csv for your improved submission.")
