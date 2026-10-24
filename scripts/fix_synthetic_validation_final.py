import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Import the pandas compatibility fix
try:
    from utils.pandas_compat import fix_pandas_compatibility, patch_seaborn
    # Apply the fixes before importing seaborn
    restore_pandas = fix_pandas_compatibility()
    patch_seaborn()
    import seaborn as sns
    print("Applied pandas compatibility fixes")
except ImportError:
    # Fall back to matplotlib if the fixes aren't available
    print("Could not apply pandas compatibility fixes - using matplotlib only")
    import matplotlib.pyplot as plt

def analyze_id_mismatch():
    # ... existing code from fix_synthetic_validation_fixed2.py ...
    # (Keep this function as is)
    return final_submission, synthetic_df, offset

def create_aligned_synthetic_data(submission_df, synthetic_df, offset=None):
    # ... existing code from fix_synthetic_validation_fixed2.py ...
    # (Keep this function as is)
    return aligned_df

def validate_with_aligned_data():
    """Validate submission using aligned synthetic data"""
    print("\n=== VALIDATING WITH ALIGNED SYNTHETIC DATA ===")
    
    # Load submission and synthetic data
    final_submission, synthetic_df, offset = analyze_id_mismatch()
    
    if final_submission is None or synthetic_df is None:
        print("Cannot proceed with validation due to data loading issues")
        return False
    
    # Create aligned synthetic data
    aligned_df = create_aligned_synthetic_data(final_submission, synthetic_df, offset)
    
    # Double-check that arrays have the same length
    predictions = final_submission['rainfall'].values
    ground_truth = aligned_df['rainfall'].values
    
    if len(predictions) != len(ground_truth):
        print(f"ERROR: Array length mismatch - predictions: {len(predictions)}, ground_truth: {len(ground_truth)}")
        print("Cannot proceed with validation. Check alignment function.")
        return False
    
    print(f"Validation arrays verified: {len(predictions)} elements each")
    
    # Calculate metrics
    try:
        auc = roc_auc_score(ground_truth > 0.5, predictions)  # Convert to binary for AUC
        print(f"AUC: {auc:.6f}")
    except Exception as e:
        print(f"Could not calculate AUC: {e}")
        auc = None
    
    # Correlation
    try:
        correlation = np.corrcoef(ground_truth, predictions)[0, 1]
        print(f"Correlation: {correlation:.6f}")
    except Exception as e:
        print(f"Could not calculate correlation: {e}")
        correlation = None
    
    # Create visualization - using more robust plotting that avoids seaborn KDE issues
    os.makedirs('analysis/validation', exist_ok=True)
    fig = plt.figure(figsize=(12, 8))
    
    # Scatter plot of predictions vs ground truth
    plt.subplot(2, 1, 1)
    plt.scatter(predictions, ground_truth, alpha=0.3)
    plt.xlabel('Predictions')
    plt.ylabel('Synthetic Ground Truth')
    plt.title('Predictions vs Aligned Synthetic Data')
    
    # Histogram instead of KDE plot to avoid pandas config issues
    plt.subplot(2, 1, 2)
    plt.hist(predictions, bins=30, alpha=0.5, label='Predictions', density=True)
    plt.hist(ground_truth, bins=30, alpha=0.5, label='Synthetic Ground Truth', density=True)
    plt.legend()
    plt.title('Distribution Comparison')
    
    plt.tight_layout()
    plt.savefig('analysis/validation/aligned_validation_fixed.png')
    print("Saved validation visualization to analysis/validation/aligned_validation_fixed.png")
    
    # Performance assessment
    print("\n=== PERFORMANCE ASSESSMENT ===")
    
    if correlation is not None and correlation > 0.7 or (auc is not None and auc > 0.85):
        print("EXCELLENT - Your model is performing very well!")
    elif correlation is not None and correlation > 0.5 or (auc is not None and auc > 0.75):
        print("GOOD - Your model is showing strong performance")
    else:
        print("NEEDS IMPROVEMENT - Consider refining your model approach")
        
    # Also save numeric metrics to a file
    metrics = {
        'AUC': auc if auc is not None else 0.0,
        'Correlation': correlation if correlation is not None else 0.0,
        'Common IDs': len(predictions)
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('analysis/validation/metrics.csv', index=False)
    print("Saved metrics to analysis/validation/metrics.csv")
    
    return True

if __name__ == "__main__":
    print("Fixing synthetic validation and validating final submission...")
    validate_with_aligned_data()
    print("\nValidation complete!")
