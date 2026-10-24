import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error
import os

def train_simple_uncertainty_model():
    """Simple uncertainty-aware model without specialized libraries"""
    print("Training a simple uncertainty-aware model...")
    
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
    test_preds = np.zeros(len(X_test))
    
    # Initialize arrays for uncertainty estimation
    oof_uncertainty = np.zeros(len(X))
    test_uncertainty = np.zeros(len(X_test))
    
    # Directory for saving models
    os.makedirs('models/simple_uncertainty', exist_ok=True)
    
    fold_scores = []
    
    # Import models - try a simpler approach using sklearn models
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    
    # Train on folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\nTraining fold {fold+1}/5")
        
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values
        
        # Train a base model
        base_model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42+fold
        )
        
        base_model.fit(X_train, y_train)
        
        # Train a second model to predict the errors of the first model
        base_preds_train = base_model.predict(X_train)
        errors = np.abs(base_preds_train - y_train)
        
        # Train error model
        error_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=5,
            random_state=42+fold
        )
        
        error_model.fit(X_train, errors)
        
        # Predict for validation set
        base_preds_val = base_model.predict(X_val)
        uncertainty_val = error_model.predict(X_val)
        
        oof_preds[val_idx] = base_preds_val
        oof_uncertainty[val_idx] = uncertainty_val
        
        # Predict for test set
        base_preds_test = base_model.predict(X_test_scaled)
        uncertainty_test = error_model.predict(X_test_scaled)
        
        test_preds += base_preds_test / 5
        test_uncertainty += uncertainty_test / 5
        
        # Calculate metrics
        auc = roc_auc_score(y_val, base_preds_val)
        rmse = np.sqrt(mean_squared_error(y_val, base_preds_val))
        fold_scores.append(auc)
        
        print(f"Fold {fold+1} - AUC: {auc:.6f}, RMSE: {rmse:.6f}")
        
        # Create uncertainty visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(base_preds_val, y_val, alpha=0.5, c=uncertainty_val, cmap='viridis')
        plt.colorbar(label='Prediction Uncertainty')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Fold {fold+1} Predictions with Uncertainty')
        plt.savefig(f'models/simple_uncertainty/uncertainty_fold_{fold}.png')
    
    # Calculate overall AUC
    cv_auc = roc_auc_score(y, oof_preds)
    print(f"\nCV AUC: {cv_auc:.6f}")
    print(f"Fold AUCs: {[round(score, 6) for score in fold_scores]}")
    
    # Create uncertainty plots - using matplotlib directly to avoid seaborn issues
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Scatter plot with uncertainty as color
    plt.subplot(1, 2, 1)
    plt.scatter(oof_preds, y, alpha=0.5, c=oof_uncertainty, cmap='viridis')
    plt.colorbar(label='Uncertainty')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('CV Predictions with Uncertainty')
    
    # Plot 2: Histogram of uncertainties using matplotlib
    plt.subplot(1, 2, 2)
    plt.hist(oof_uncertainty, bins=30, alpha=0.7, color='blue')
    plt.title('Distribution of Prediction Uncertainty')
    plt.xlabel('Uncertainty')
    
    plt.tight_layout()
    plt.savefig('models/simple_uncertainty/uncertainty_analysis.png')
    
    # Create uncertainty-weighted AUC
    confidence_weights = 1 / (1 + oof_uncertainty)
    confidence_weights /= confidence_weights.sum()
    
    # Measure weighted AUC
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, oof_preds, sample_weight=confidence_weights)
    weighted_auc = np.trapz(tpr, fpr)
    print(f"Uncertainty-weighted AUC: {weighted_auc:.6f}")
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'rainfall': test_preds
    })
    
    # Save full submission with uncertainty
    detailed_submission = pd.DataFrame({
        'id': test_df['id'],
        'rainfall': test_preds,
        'uncertainty': test_uncertainty
    })
    
    uncertainty_path = 'data/submission_with_uncertainty.csv'
    detailed_submission.to_csv(uncertainty_path, index=False)
    print(f"Saved detailed submission to {uncertainty_path}")
    
    # Save standard submission
    submission_path = 'data/submission_uncertainty_model.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}")
    
    # Copy to keep directory
    os.makedirs('data/keep', exist_ok=True)
    import shutil
    keep_path = os.path.join('data/keep', 'submission_uncertainty_model.csv')
    shutil.copy2(submission_path, keep_path)
    print(f"Saved copy to {keep_path}")
    
    # Validate the submission if possible
    try:
        from validate_submission import validate_submission
        print("\nValidating submission...")
        result = validate_submission(submission_path)
        if result and 'estimated_score' in result:
            print(f"Estimated score: {result['estimated_score']:.6f}")
    except Exception as e:
        print(f"Could not validate submission: {str(e)}")
    
    # Create high-confidence submission
    high_conf_threshold = np.percentile(test_uncertainty, 75)
    high_conf_mask = test_uncertainty <= high_conf_threshold
    print(f"Creating high-confidence submission with {high_conf_mask.sum()} predictions")
    
    high_conf_submission = pd.DataFrame({
        'id': test_df['id'].values[high_conf_mask],
        'rainfall': test_preds[high_conf_mask]
    })
    
    high_conf_path = 'data/submission_high_confidence.csv'
    high_conf_submission.to_csv(high_conf_path, index=False)
    print(f"Saved high-confidence submission to {high_conf_path}")
    
    return {
        'cv_auc': cv_auc,
        'weighted_auc': weighted_auc,
        'oof_preds': oof_preds,
        'oof_uncertainty': oof_uncertainty,
        'test_preds': test_preds,
        'test_uncertainty': test_uncertainty
    }

if __name__ == "__main__":
    print("Training simple uncertainty-aware model...")
    results = train_simple_uncertainty_model()
    print("Training complete!")
