import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
import warnings
from calibrated_validation import validate_calibrated

warnings.filterwarnings('ignore')

def find_best_candidate():
    """use calibrated_validation.py to score submissions in /data/candidates"""
    print("Searching /data/candidates ...")
    
    # Find all submissions in /data/candidates
    submission_files = glob.glob(os.path.join('data/candidates', '*.csv'))
    
    if not submission_files:
        print("No submission files, error")
        return
    
    print(f"Found {len(submission_files)} submission files")
    print("\nValidating files...")

    # Retain all submission paths and validation scores
    results = []
    
    # Validate all
    for submission_path in submission_files:
        try:
            validation = validate_calibrated(submission_path)
            score = validation['predicted_score']
            results.append({'path': submission_path, 'score': score})
            print(f"{submission_path} validation score: {score:.6f}")
        except Exception as e:
            print(f"Could not validate {submission_path}: {str(e)}")
    
    # sort results from high score to low score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    for i, result in enumerate(results[:5]):  # Print top 5
        print(f"{i+1}. {result['path']} - Score: {result['score']:.6f}")
    
    # also save results to file
    # Convert results to DataFrame for easier saving
    results_df = pd.DataFrame(results)

    # Save to CSV file
    results_file = "submission_results.csv"
    results_df.to_csv(results_file, index=False)

    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    find_best_candidate()
