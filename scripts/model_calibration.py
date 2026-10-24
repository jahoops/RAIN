import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve

def analyze_score_discrepancy(cv_score, submission_score, strategy='minimal'):
    """
    Analyze the discrepancy between cross-validation and submission scores
    
    Args:
        cv_score: Cross-validation score (e.g., 0.862)
        submission_score: Actual submission score (e.g., 0.832)
        strategy: Feature engineering strategy used
    """
    print(f"\nAnalyzing score discrepancy for {strategy} strategy:")
    print(f"CV Score: {cv_score:.6f}")
    print(f"Submission Score: {submission_score:.6f}")
    print(f"Discrepancy: {cv_score - submission_score:.6f} ({(cv_score - submission_score)/cv_score*100:.2f}%)")
    
    # Load OOF predictions if available
    oof_path = 'models/oof_predictions.csv'
    if os.path.exists(oof_path):
        oof_df = pd.read_csv(oof_path)
        
        # Plot histogram of predictions to check distribution
        plt.figure(figsize=(10, 6))
        plt.hist(oof_df['prediction'], bins=50, alpha=0.5, label='Predictions')
        plt.axvline(0.5, color='red', linestyle='--', label='Decision Threshold')
        plt.xlabel('Prediction Value')
        plt.ylabel('Count')
        plt.title('Distribution of Model Predictions')
        plt.legend()
        plt.savefig(f'models/{strategy}_prediction_distribution.png')
        
        # Generate calibration curve
        y_true = oof_df['true_value']
        y_pred = oof_df['prediction']
        
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
        
        plt.figure(figsize=(10, 6))
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.savefig(f'models/{strategy}_calibration_curve.png')
        
        # Calculate Brier score (lower is better)
        brier_score = np.mean((y_pred - y_true) ** 2)
        print(f"Brier Score: {brier_score:.6f}")
        
    # Potential reasons for discrepancy
    print("\nPotential reasons for the score discrepancy:")
    print("1. Overfitting to the training data")
    print("2. Distribution shift between training and test data")
    print("3. Temporal effects if the data has a time component")
    print("4. Random variation in the data")
    
    # Recommendations
    print("\nRecommendations to improve model robustness:")
    print("1. Use more aggressive regularization (higher alpha/lambda values)")
    print("2. Consider simpler models with fewer features")
    print("3. Implement Bayesian optimization for hyperparameter tuning")
    print("4. Try model ensembling with different algorithms")
    print("5. Use a more robust cross-validation strategy (e.g., nested CV)")

if __name__ == "__main__":
    # Analyze the discrepancy for the minimal model
    analyze_score_discrepancy(0.862, 0.832, 'minimal')
    
    # You can also analyze other models if you have their scores
    # analyze_score_discrepancy(0.875, 0.840, 'all')
    # analyze_score_discrepancy(0.870, 0.835, 'boruta_top')
