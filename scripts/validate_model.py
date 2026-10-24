import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import pickle
import argparse

def evaluate_model(model, X, y, cv=5):
    """
    Evaluate model using cross-validation with ROC AUC metric
    
    Args:
        model: Trained model with predict_proba method
        X: Features dataframe
        y: Target series
        cv: Number of cross-validation folds
    
    Returns:
        Mean ROC AUC score across folds
    """
    # Use StratifiedKFold to maintain class distribution
    cv_folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Calculate cross-validated ROC AUC scores
    auc_scores = cross_val_score(
        model, X, y, 
        cv=cv_folds,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    print(f"Cross-validated ROC AUC scores: {auc_scores}")
    print(f"Mean ROC AUC: {np.mean(auc_scores):.4f} (±{np.std(auc_scores):.4f})")
    
    return np.mean(auc_scores)

def main():
    parser = argparse.ArgumentParser(description='Validate model using ROC AUC')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to validation features')
    parser.add_argument('--target_path', type=str, required=True, help='Path to validation targets')
    parser.add_argument('--cv', type=int, default=5, help='Number of cross-validation folds')
    
    args = parser.parse_args()
    
    # Load model
    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load data
    X = pd.read_csv(args.data_path)
    y = pd.read_csv(args.target_path).iloc[:, 0]
    
    # Evaluate model
    evaluate_model(model, X, y, args.cv)

if __name__ == '__main__':
    main()
