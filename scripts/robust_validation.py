import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import os
from data_loader import load_data, preprocess_data
from feature_engineering import engineer_features

def plot_distribution_comparison(train_df, test_df, columns=None):
    """Compare distributions between train and test data"""
    print("Examining train vs. test distribution differences...")
    
    if columns is None:
        # Select numeric columns that appear in both datasets
        columns = [col for col in train_df.columns 
                  if col in test_df.columns 
                  and col not in ['id', 'rainfall']
                  and np.issubdtype(train_df[col].dtype, np.number)]
        
        # Limit to 10 columns for visualization
        if len(columns) > 10:
            columns = columns[:10]
    
    # Create plot
    n_cols = len(columns)
    fig, axes = plt.subplots(n_cols, 1, figsize=(12, 4*n_cols))
    
    if n_cols == 1:
        axes = [axes]  # Make sure axes is always a list
    
    for i, col in enumerate(columns):
        # Calculate KL divergence as a measure of distribution difference
        # Using histograms for approximation
        hist_train, bin_edges = np.histogram(train_df[col].dropna(), bins=30, density=True)
        hist_test, _ = np.histogram(test_df[col].dropna(), bins=bin_edges, density=True)
        
        # Replace zeros to avoid division by zero
        hist_train = np.where(hist_train == 0, 1e-10, hist_train)
        hist_test = np.where(hist_test == 0, 1e-10, hist_test)
        
        # Calculate KL divergence: KL(P||Q) = sum(P * log(P/Q))
        kl_div = np.sum(hist_train * np.log(hist_train / hist_test)) if not np.any(hist_test == 0) else np.inf
        
        # Plot distributions
        axes[i].hist(train_df[col].dropna(), bins=30, alpha=0.5, density=True, label='Train')
        axes[i].hist(test_df[col].dropna(), bins=30, alpha=0.5, density=True, label='Test')
        axes[i].set_title(f"{col} (KL Div: {kl_div:.4f})")
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('/home/USER/Projects/RAIN/plots/train_test_distributions.png')
    print("Distribution comparison plot saved")
    return columns

def create_robust_validation_folds(df, day_col='day', n_folds=5):
    """Create time-based validation folds with gap between train and test"""
    if day_col not in df.columns:
        raise ValueError(f"Day column '{day_col}' not found in dataframe")
    
    # Sort by day
    df = df.sort_values(day_col)
    
    # Get unique days
    unique_days = df[day_col].unique()
    
    # Calculate fold size
    fold_size = len(unique_days) // n_folds
    
    # Create folds with gaps
    gap_size = fold_size // 5  # 20% gap between train and validation
    
    folds = []
    for i in range(n_folds - 1):  # Create n_folds-1 train/val splits
        # Calculate fold boundaries
        train_end_idx = (i + 1) * fold_size
        val_start_idx = train_end_idx + gap_size
        val_end_idx = val_start_idx + fold_size
        
        # Get days for train and validation
        train_days = unique_days[:train_end_idx]
        val_days = unique_days[val_start_idx:min(val_end_idx, len(unique_days))]
        
        # Create boolean masks
        train_mask = df[day_col].isin(train_days)
        val_mask = df[day_col].isin(val_days)
        
        folds.append((train_mask, val_mask))
    
    return folds

def evaluate_with_future_gap(train_df, test_df, gap_days=30, feature_strategy='all_with_patterns'):
    """Evaluate model with a time gap to simulate future prediction scenario"""
    print(f"\nEvaluating model with {gap_days}-day future gap...")
    
    if 'day' not in train_df.columns:
        print("No 'day' column found - cannot perform future gap validation")
        return None
    
    # Sort by day
    train_df = train_df.sort_values('day')
    
    # Apply feature engineering
    train_df, test_df = engineer_features(train_df, test_df, strategy=feature_strategy)
    
    # Find cutoff day for validation
    max_day = train_df['day'].max()
    cutoff_day = max_day - gap_days
    
    # Split data ensuring no data leakage
    train_mask = train_df['day'] <= cutoff_day
    val_mask = train_df['day'] > cutoff_day
    
    X_train = train_df.loc[train_mask].drop(['id', 'rainfall', 'day'], axis=1, errors='ignore')
    y_train = train_df.loc[train_mask, 'rainfall']
    
    X_val = train_df.loc[val_mask].drop(['id', 'rainfall', 'day'], axis=1, errors='ignore')
    y_val = train_df.loc[val_mask, 'rainfall']
    
    print(f"Train data: {X_train.shape[0]} samples, days 1-{cutoff_day}")
    print(f"Validation data: {X_val.shape[0]} samples, days {cutoff_day+1}-{max_day}")
    
    # Train model with more regularization to combat overfitting
    model = xgb.XGBRegressor(
        n_estimators=300, 
        learning_rate=0.03,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=3.0,
        reg_lambda=3.0,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    val_preds_raw = model.predict(X_val)
    
    # Evaluate uncalibrated predictions
    uncalibrated_score = roc_auc_score(y_val, val_preds_raw)
    print(f"Uncalibrated future gap validation score: {uncalibrated_score:.6f}")
    
    # Apply isotonic calibration
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(val_preds_raw.reshape(-1, 1), y_val)
    
    # Make calibrated predictions for test set
    X_test = test_df.drop(['id'], axis=1, errors='ignore')
    test_preds_raw = model.predict(X_test)
    test_preds_calibrated = iso.predict(test_preds_raw.reshape(-1, 1))
    
    # Create submission with calibrated predictions
    submission = pd.DataFrame({
        'id': test_df['id'],
        'rainfall': test_preds_calibrated
    })
    
    submission_path = f'/home/USER/Projects/RAIN/data/future_gap_{gap_days}d_submission.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Calibrated submission saved to {submission_path}")
    
    # Plot predictions vs actual for validation set
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, val_preds_raw, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted - Future Gap Validation')
    plt.savefig('/home/USER/Projects/RAIN/plots/future_gap_predictions.png')
    
    return uncalibrated_score

def train_with_ensemble_and_stacking():
    """Use ensemble of multiple models with stacking for more robust predictions"""
    print("\nTraining ensemble with stacking...")
    
    # Load and process data
    train_df, test_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # Apply best feature engineering strategy
    train_df, test_df = engineer_features(train_df, test_df, strategy='all_with_patterns')
    
    # Create robust validation folds
    if 'day' in train_df.columns:
        folds = create_robust_validation_folds(train_df, day_col='day', n_folds=5)
        print(f"Created {len(folds)} time-based validation folds with gaps")
    else:
        # Fall back to regular KFold if no day column
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        folds = [(train_idx, val_idx) for train_idx, val_idx in 
                kf.split(train_df.drop(['id', 'rainfall'], axis=1, errors='ignore'))]
    
    # Prepare data
    X = train_df.drop(['id', 'rainfall'], axis=1, errors='ignore')
    y = train_df['rainfall']
    X_test = test_df.drop(['id'], axis=1, errors='ignore')
    
    # Define base models
    models = [
        ('xgb', xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4,
                                subsample=0.7, colsample_bytree=0.7,
                                reg_alpha=2.0, reg_lambda=2.0, random_state=42)),
        ('lgb', lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, max_depth=4,
                                subsample=0.7, colsample_bytree=0.7,
                                reg_alpha=2.0, reg_lambda=2.0, random_state=42)),
    ]
    
    # Train first-level models
    oof_preds = np.zeros((X.shape[0], len(models)))
    test_preds = np.zeros((X_test.shape[0], len(models)))
    
    for fold_idx, (train_mask, val_mask) in enumerate(folds):
        print(f"\nTraining fold {fold_idx+1}/{len(folds)}...")
        
        # Get fold data
        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]
        X_val = X.loc[val_mask]
        y_val = y.loc[val_mask]
        
        for model_idx, (name, model) in enumerate(models):
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Save predictions
            oof_preds[val_mask, model_idx] = model.predict(X_val)
            test_preds[:, model_idx] += model.predict(X_test) / len(folds)
            
            # Evaluate
            fold_score = roc_auc_score(y_val, oof_preds[val_mask, model_idx])
            print(f"{name} fold {fold_idx+1} score: {fold_score:.6f}")
    
    # Evaluate base models
    for model_idx, (name, _) in enumerate(models):
        model_score = roc_auc_score(y, oof_preds[:, model_idx])
        print(f"\n{name} overall CV score: {model_score:.6f}")
    
    # Train meta-model (stacking)
    meta_model = LogisticRegression()
    meta_model.fit(oof_preds, y)
    
    # Make final predictions
    final_preds = meta_model.predict_proba(test_preds)[:, 1]
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'rainfall': final_preds
    })
    
    submission_path = '/home/USER/Projects/RAIN/data/stacked_ensemble_submission.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Stacked ensemble submission saved to {submission_path}")
    
    # Save meta-model weights for analysis
    print("\nMeta-model coefficients (model importance):")
    for (name, _), coef in zip(models, meta_model.coef_[0]):
        print(f"{name}: {coef:.6f}")

def main():
    """Run comprehensive validation and create more robust submissions"""
    print("Starting robust validation process...")
    
    # Create plots directory if it doesn't exist
    os.makedirs('/home/USER/Projects/RAIN/plots', exist_ok=True)
    
    # Load and preprocess data
    train_df, test_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # Compare distributions
    plot_distribution_comparison(train_df, test_df)
    
    # Evaluate with future gap
    gap_scores = []
    for gap in [15, 30, 45]:
        score = evaluate_with_future_gap(train_df.copy(), test_df.copy(), gap_days=gap)
        if score is not None:
            gap_scores.append((gap, score))
    
    # Find best gap
    if gap_scores:
        best_gap = min(gap_scores, key=lambda x: abs(x[1] - 0.74))[0]
        print(f"\nBest gap ({best_gap} days) produced a score closest to the known submission score of 0.74")
        print("Re-running with this optimal gap to generate final submission...")
        evaluate_with_future_gap(train_df.copy(), test_df.copy(), gap_days=best_gap)
    
    # Train stacked ensemble
    train_with_ensemble_and_stacking()
    
    print("\nCompleted robust validation process.")
    print("Please try the submissions in this order of preference:")
    print("1. future_gap_<best_gap>d_submission.csv")
    print("2. stacked_ensemble_submission.csv")
    print("3. calibrated_submission.csv")

if __name__ == "__main__":
    main()
