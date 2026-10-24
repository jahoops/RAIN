import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_squared_error
import seaborn as sns

def validate_against_synthetic():
    """Validate the final submission against synthetic data"""
    print("=== VALIDATING FINAL SUBMISSION AGAINST SYNTHETIC DATA ===")
    
    # Load the final submission
    final_path = 'data/final/submission_final.csv'
    if not os.path.exists(final_path):
        final_path = 'data/final/submission_final_improved.csv'
        if not os.path.exists(final_path):
            print("ERROR: Could not find final submission file!")
            return False
    
    final_submission = pd.read_csv(final_path)
    print(f"Loaded final submission from {final_path}")
    print(f"Contains {len(final_submission)} predictions")
    
    # Find synthetic data
    synthetic_paths = [
        'data/synthetic_validation.csv',
        'data/synthetic_bootstrap.csv',
        'data/synthetic_data.csv',
        'data/synthetic_test.csv'
    ]
    
    synthetic_df = None
    for path in synthetic_paths:
        if os.path.exists(path):
            synthetic_df = pd.read_csv(path)
            print(f"Found synthetic data at {path}")
            break
    
    if synthetic_df is None:
        print("ERROR: Could not find synthetic data for validation!")
        print("Checking if validation module exists...")
        
        try:
            from validate_submission import validate_submission
            print("Found validation module, trying to validate submission...")
            result = validate_submission(final_path)
            if result and 'estimated_score' in result:
                print(f"Validation successful!")
                print(f"Estimated score: {result['estimated_score']:.6f}")
                return True
        except Exception as e:
            print(f"Could not validate using module: {str(e)}")
            return False
    
    # Find common IDs
    common_ids = set(final_submission['id']).intersection(set(synthetic_df['id']))
    print(f"Found {len(common_ids)} common IDs between submission and synthetic data")
    
    if len(common_ids) == 0:
        print("ERROR: No common IDs found. Cannot validate!")
        return False
    
    # Filter to common IDs and sort
    sub_filtered = final_submission[final_submission['id'].isin(common_ids)].sort_values('id')
    syn_filtered = synthetic_df[synthetic_df['id'].isin(common_ids)].sort_values('id')
    
    # Ensure they're in the same order
    assert all(sub_filtered['id'].values == syn_filtered['id'].values), "ID mismatch!"
    
    # Calculate metrics
    predictions = sub_filtered['rainfall'].values
    ground_truth = syn_filtered['rainfall'].values
    
    # If ground truth is binary but predictions are not, this may be a regression vs classification mismatch
    if set(np.unique(ground_truth)) == {0, 1} and not set(np.unique(predictions)).issubset({0, 1}):
        print("Note: Ground truth is binary but predictions are continuous - treating as probabilities")
    
    # AUC (for classification)
    try:
        auc = roc_auc_score(ground_truth, predictions)
        print(f"AUC: {auc:.6f}")
    except:
        print("Could not calculate AUC (may not be a classification task)")
    
    # MSE (for regression)
    mse = mean_squared_error(ground_truth, predictions)
    print(f"MSE: {mse:.6f}")
    
    # Correlation
    correlation = np.corrcoef(ground_truth, predictions)[0, 1]
    print(f"Correlation: {correlation:.6f}")
    
    # Create validation plots
    os.makedirs('analysis/validation', exist_ok=True)
    plt.figure(figsize=(15, 10))
    
    # Scatter plot of predictions vs ground truth
    plt.subplot(2, 2, 1)
    plt.scatter(predictions, ground_truth, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Predictions')
    plt.ylabel('Ground Truth')
    plt.title('Predictions vs Ground Truth')
    
    # Distribution of predictions
    plt.subplot(2, 2, 2)
    sns.histplot(predictions, label='Predictions', alpha=0.6, color='blue')
    sns.histplot(ground_truth, label='Ground Truth', alpha=0.6, color='red')
    plt.legend()
    plt.title('Distribution Comparison')
    
    # Residuals
    plt.subplot(2, 2, 3)
    residuals = ground_truth - predictions
    sns.histplot(residuals)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Residuals')
    
    # Prediction vs Error
    plt.subplot(2, 2, 4)
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predictions')
    plt.ylabel('Error')
    plt.title('Prediction vs Error')
    
    plt.tight_layout()
    plt.savefig('analysis/validation/final_submission_validation.png')
    print("Saved validation plots to analysis/validation/final_submission_validation.png")
    
    # Performance assessment
    benchmark = 0.85  # Example benchmark
    
    print("\n=== PERFORMANCE ASSESSMENT ===")
    
    estimated_score = None
    if auc >= 0.9:
        performance_tier = "OUTSTANDING - Well above benchmark!"
        estimated_score = auc
    elif auc >= 0.85:
        performance_tier = "EXCELLENT - On par with benchmark!"
        estimated_score = auc
    elif auc >= 0.80:
        performance_tier = "VERY GOOD - Close to benchmark"
        estimated_score = auc
    else:
        performance_tier = "GOOD - Room for improvement"
        estimated_score = auc
    
    print(f"Estimated contest score: {estimated_score:.6f}")
    print(f"Performance tier: {performance_tier}")
    
    return True

if __name__ == "__main__":
    print("Validating final submission...")
    success = validate_against_synthetic()
    
    if success:
        print("\nValidation complete!")
    else:
        print("\nValidation failed. Please check issues above.")
    
    # Regardless of validation success, make sure final submission is properly formatted
    final_path = 'data/final/submission_final.csv'
    if os.path.exists(final_path):
        df = pd.read_csv(final_path)
        
        # Check for required columns
        if 'id' not in df.columns or 'rainfall' not in df.columns:
            print("ERROR: Final submission missing required columns!")
        elif df['rainfall'].isna().any():
            print("ERROR: Final submission contains NaN values!")
        else:
            print("Final submission format appears valid.")
            print(f"Contains {len(df)} predictions with range {df['rainfall'].min():.4f} to {df['rainfall'].max():.4f}")
