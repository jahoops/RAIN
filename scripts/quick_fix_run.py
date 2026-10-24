import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import os
import pickle

def load_prepare_data():
    """Load and prepare data with consistent preprocessing"""
    from data_loader import load_data, preprocess_data
    from feature_engineering import engineer_features
    
    # Load data
    train_df, test_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # Apply feature engineering
    train_df, test_df = engineer_features(train_df, test_df, strategy='all')
    
    # Store IDs and target
    train_ids = train_df['id']
    test_ids = test_df['id']
    y = train_df['rainfall']
    
    # Get features only
    X = train_df.drop(['id', 'rainfall'], axis=1)
    X_test = test_df.drop(['id'], axis=1)
    
    return X, y, X_test, train_ids, test_ids

def ensure_consistency(X_train, X_test):
    """Ensure X_train and X_test have identical features"""
    # Get common columns
    common_cols = list(set(X_train.columns).intersection(set(X_test.columns)))
    
    # Use only common columns for both
    X_train = X_train[common_cols].copy()
    X_test = X_test[common_cols].copy()
    
    # Verify they have the same shape
    assert X_train.shape[1] == X_test.shape[1], "Column count mismatch!"
    assert all(X_train.columns == X_test.columns), "Column order mismatch!"
    
    print(f"Feature consistency verified: using {len(common_cols)} features")
    return X_train, X_test

def add_safe_features(X_train, X_test):
    """Add features guaranteed to be consistent"""
    # Create a combined dataset for transformation
    n_train = X_train.shape[0]
    combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    
    # Apply transformations to combined data
    for col in combined.columns:
        # Skip categorical columns
        if not np.issubdtype(combined[col].dtype, np.number):
            continue
            
        # Add squares
        combined[f"{col}_squared"] = combined[col] ** 2
        
        # Add log for positive columns
        if combined[col].min() >= 0:
            combined[f"{col}_log"] = np.log1p(combined[col])
    
    # Split back into train and test
    X_train_enhanced = combined.iloc[:n_train].copy()
    X_test_enhanced = combined.iloc[n_train:].copy()
    
    print(f"Feature count: {X_train.shape[1]} → {X_train_enhanced.shape[1]}")
    return X_train_enhanced, X_test_enhanced

def train_simple_model(X_train, y, X_test, test_ids, n_folds=5):
    """Train a simple ensemble model with XGBoost and LightGBM"""
    print(f"Training with {X_train.shape[1]} features")
    
    # Set up cross validation
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize predictions
    oof_preds_xgb = np.zeros(len(X_train))
    test_preds_xgb = np.zeros(len(X_test))
    oof_preds_lgb = np.zeros(len(X_train))
    test_preds_lgb = np.zeros(len(X_test))
    
    # Track fold scores
    fold_scores = []
    
    # Train across folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
        print(f"Training fold {fold+1}/{n_folds}")
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train XGBoost with conservative parameters
        xgb_model = xgb.XGBRegressor(
            objective='binary:logistic',
            eval_metric='auc',
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42+fold
        )
        
        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Train LightGBM
        lgb_model = lgb.LGBMRegressor(
            objective='binary',
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42+fold
        )
        
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Get out-of-fold predictions
        oof_preds_xgb[val_idx] = xgb_model.predict(X_val)
        oof_preds_lgb[val_idx] = lgb_model.predict(X_val)
        
        # Get test predictions
        test_preds_xgb += xgb_model.predict(X_test) / n_folds
        test_preds_lgb += lgb_model.predict(X_test) / n_folds
        
        # Calculate fold score
        xgb_fold_score = roc_auc_score(y_val, oof_preds_xgb[val_idx])
        lgb_fold_score = roc_auc_score(y_val, oof_preds_lgb[val_idx])
        fold_scores.append((xgb_fold_score, lgb_fold_score))
        print(f"Fold {fold+1} - XGB: {xgb_fold_score:.6f}, LGB: {lgb_fold_score:.6f}")
    
    # Calculate overall CV scores
    xgb_cv = roc_auc_score(y, oof_preds_xgb)
    lgb_cv = roc_auc_score(y, oof_preds_lgb)
    
    print(f"XGBoost CV: {xgb_cv:.6f}")
    print(f"LightGBM CV: {lgb_cv:.6f}")
    
    # Create weighted ensemble
    weight_xgb = xgb_cv / (xgb_cv + lgb_cv)
    weight_lgb = lgb_cv / (xgb_cv + lgb_cv)
    ensemble_preds = (test_preds_xgb * weight_xgb) + (test_preds_lgb * weight_lgb)
    
    # Create submissions
    ensemble_sub = pd.DataFrame({'id': test_ids, 'rainfall': ensemble_preds})
    xgb_sub = pd.DataFrame({'id': test_ids, 'rainfall': test_preds_xgb})
    lgb_sub = pd.DataFrame({'id': test_ids, 'rainfall': test_preds_lgb})
    
    # Save submissions
    os.makedirs('data/keep', exist_ok=True)
    ensemble_sub.to_csv('data/submission_ensemble_quick.csv', index=False)
    xgb_sub.to_csv('data/submission_xgb_quick.csv', index=False)
    lgb_sub.to_csv('data/submission_lgb_quick.csv', index=False)
    
    print("Saved submissions:")
    print("- data/submission_ensemble_quick.csv")
    print("- data/submission_xgb_quick.csv")
    print("- data/submission_lgb_quick.csv")
    
    # Validate with synthetic data if available
    try:
        from validate_submission import validate_submission
        print("\nValidating ensemble submission against synthetic data...")
        validate_submission('data/submission_ensemble_quick.csv', quiet=False)
    except Exception as e:
        print(f"Could not validate with synthetic data: {e}")
    
    return {
        'xgb_cv': xgb_cv,
        'lgb_cv': lgb_cv,
        'ensemble_preds': ensemble_preds
    }

def run_quick_fix():
    """Run a simplified, reliable model to avoid feature mismatch issues"""
    print("\n===== RUNNING QUICK FIX MODEL =====\n")
    
    # Load data
    X, y, X_test, train_ids, test_ids = load_prepare_data()
    
    # Ensure feature consistency
    X, X_test = ensure_consistency(X, X_test)
    
    # Add safe features
    X_enhanced, X_test_enhanced = add_safe_features(X, X_test)
    
    # Verify final consistency
    assert X_enhanced.shape[1] == X_test_enhanced.shape[1], "Feature count mismatch!"
    print(f"Final feature count: {X_enhanced.shape[1]}")
    
    # Train model
    results = train_simple_model(X_enhanced, y, X_test_enhanced, test_ids)

if __name__ == "__main__":
    run_quick_fix()
