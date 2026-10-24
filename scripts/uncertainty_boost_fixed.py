import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import ngboost
from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error
import matplotlib.pyplot as plt
import os

def train_ngboost_model():
    """Train NGBoost model with uncertainty quantification"""
    print("Training NGBoost model with uncertainty quantification...")
    
    # Load data
    try:
        from data_loader import load_data, preprocess_data
        train_df, test_df = load_data()
        train_df, test_df = preprocess_data(train_df, test_df)
    except:
        print("Could not load raw data. Trying to load advanced features...")
        try:
            train_df = pd.read_csv('data/train_advanced_features.csv')
            test_df = pd.read_csv('data/test_advanced_features.csv')
        except:
            print("No data files found. Please check data directory.")
            return None
    
    # Prepare features and target
    X = train_df.drop(['id', 'rainfall'], axis=1)
    y = train_df['rainfall']
    X_test = test_df.drop(['id'], axis=1)
    
    # Ensure feature consistency
    common_cols = list(set(X.columns).intersection(set(X_test.columns)))
    X = X[common_cols]
    X_test = X_test[common_cols]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    # Set up cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize arrays for predictions
    oof_preds = np.zeros(len(X))
    oof_stds = np.zeros(len(X))  # For uncertainty
    test_preds = np.zeros(len(X_test))
    test_stds = np.zeros(len(X_test))  # For uncertainty
    
    # Directory for saving models
    os.makedirs('models/ngboost', exist_ok=True)
    
    fold_scores = []
    
    # Train on folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\nTraining fold {fold+1}/5")
        
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values
        
        # Build and train NGBoost model
        ngb = ngboost.NGBRegressor(
            Dist=Normal,
            Score=LogScore,
            n_estimators=500,
            learning_rate=0.02,
            natural_gradient=True,
            verbose=False,
            random_state=42+fold
        )
        
        ngb.fit(X_train, y_train, X_val=X_val, Y_val=y_val, early_stopping_rounds=20)
        
        # Make predictions with uncertainty
        preds_dist = ngb.pred_dist(X_val)
        oof_preds[val_idx] = preds_dist.loc
        oof_stds[val_idx] = preds_dist.scale
        
        # Make test predictions
        test_dist = ngb.pred_dist(X_test_scaled)
        test_fold_preds = test_dist.loc
        test_fold_stds = test_dist.scale
        
        test_preds += test_fold_preds / 5
        test_stds += test_fold_stds / 5
        
        # Calculate metrics
        auc = roc_auc_score(y_val, oof_preds[val_idx])
        rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        fold_scores.append(auc)
        
        print(f"Fold {fold+1} - AUC: {auc:.6f}, RMSE: {rmse:.6f}")
        
        # Create uncertainty visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(oof_preds[val_idx], y_val, alpha=0.5, 
                   c=oof_stds[val_idx], cmap='viridis')
        plt.colorbar(label='Prediction Uncertainty (std)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Fold {fold+1} Predictions with Uncertainty')
        plt.savefig(f'models/ngboost/uncertainty_fold_{fold}.png')
    
    # Calculate overall AUC
    cv_auc = roc_auc_score(y, oof_preds)
    print(f"\nCV AUC: {cv_auc:.6f}")
    print(f"Fold AUCs: {[round(score, 6) for score in fold_scores]}")
    
    # Replace problematic seaborn histplot with direct matplotlib histogram
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Scatter plot of predictions vs actual, colored by uncertainty
    plt.subplot(1, 2, 1)
    plt.scatter(oof_preds, y, alpha=0.5, c=oof_stds, cmap='viridis')
    plt.colorbar(label='Prediction Uncertainty (std)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('CV Predictions with Uncertainty')
    
    # Plot 2: Histogram of uncertainties (using matplotlib instead of seaborn)
    plt.subplot(1, 2, 2)
    plt.hist(oof_stds, bins=30, alpha=0.7, color='blue')
    plt.title('Distribution of Prediction Uncertainty')
    plt.xlabel('Uncertainty (std)')
    
    plt.tight_layout()
    plt.savefig('models/ngboost/uncertainty_analysis.png')
    
    # Create weighted ensemble based on uncertainty
    # Less weight to uncertain predictions
    confidence_weights = 1 / (1 + oof_stds)
    confidence_weights /= confidence_weights.sum()
    
    # Measure weighted AUC
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, oof_preds, sample_weight=confidence_weights)
    weighted_auc = np.trapz(tpr, fpr)
    print(f"Uncertainty-weighted AUC: {weighted_auc:.6f}")
    
    # Create submission with confidence intervals
    submission = pd.DataFrame({
        'id': test_df['id'],
        'rainfall': test_preds,
        'uncertainty': test_stds
    })
    
    # Save submission with uncertainty
    submission_path = 'data/submission_uncertainty_boost.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}")
    
    # Also save a version without uncertainty column for contest submission
    contest_submission = submission[['id', 'rainfall']]
    contest_path = 'data/submission_ngboost.csv'
    contest_submission.to_csv(contest_path, index=False)
    print(f"Saved contest submission to {contest_path}")
    
    # Copy to keep directory
    os.makedirs('data/keep', exist_ok=True)
    import shutil
    shutil.copy2(contest_path, os.path.join('data/keep', 'submission_ngboost.csv'))
    
    # Save uncertainty threshold analysis
    uncertainty_thresholds = np.percentile(oof_stds, [50, 75, 90, 95, 99])
    results = []
    
    for threshold in uncertainty_thresholds:
        # Calculate AUC for samples below threshold (more certain predictions)
        mask = oof_stds <= threshold
        if mask.sum() > 0:
            certain_auc = roc_auc_score(y[mask], oof_preds[mask])
            results.append({
                'threshold': threshold,
                'coverage': mask.mean() * 100,
                'count': mask.sum(),
                'auc': certain_auc
            })
    
    threshold_df = pd.DataFrame(results)
    threshold_df.to_csv('models/ngboost/uncertainty_thresholds.csv', index=False)
    print("\nUncertainty threshold analysis:")
    print(threshold_df)
    
    # Create high-confidence submission (only most confident predictions)
    high_conf_threshold = np.percentile(test_stds, 75)  # Top 75% most certain
    high_conf_mask = test_stds <= high_conf_threshold
    high_conf_ids = test_df['id'].values[high_conf_mask]
    high_conf_preds = test_preds[high_conf_mask]
    
    high_conf_submission = pd.DataFrame({
        'id': high_conf_ids,
        'rainfall': high_conf_preds
    })
    
    high_conf_path = 'data/submission_ngboost_high_confidence.csv'
    high_conf_submission.to_csv(high_conf_path, index=False)
    print(f"Saved high-confidence submission with {len(high_conf_submission)} predictions")
    
    # Validate the submission if possible
    try:
        from validate_submission import validate_submission
        print("\nValidating NGBoost submission...")
        result = validate_submission(contest_path)
        if result and 'estimated_score' in result:
            print(f"Estimated score: {result['estimated_score']:.6f}")
    except Exception as e:
        print(f"Could not validate submission: {str(e)}")
    
    return {
        'cv_auc': cv_auc,
        'fold_scores': fold_scores,
        'oof_preds': oof_preds,
        'oof_uncertainty': oof_stds,
        'test_preds': test_preds,
        'test_uncertainty': test_stds
    }

if __name__ == "__main__":
    print("Training NGBoost model with uncertainty quantification...")
    results = train_ngboost_model()
    print("Uncertainty-aware model training complete!")
