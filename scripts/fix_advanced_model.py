import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
import os
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import pickle

def load_and_prepare_data():
    """Load data with consistent preprocessing"""
    from data_loader import load_data, preprocess_data
    from feature_engineering import engineer_features
    
    # Load data
    train_df, test_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # Apply base feature engineering
    train_df, test_df = engineer_features(train_df, test_df, strategy='all')
    
    # Store original IDs
    train_ids = train_df['id']
    test_ids = test_df['id']
    
    # Extract target
    y = train_df['rainfall']
    
    # Drop non-feature columns
    X = train_df.drop(['id', 'rainfall'], axis=1)
    X_test = test_df.drop(['id'], axis=1)
    
    # Create directory for models if not exists
    os.makedirs('models/advanced', exist_ok=True)
    
    return X, y, X_test, train_ids, test_ids

def create_consistent_features(X_train, X_test):
    """Create features with consistent transformations between train and test sets"""
    print("Creating consistent advanced features...")
    
    # Make copies to avoid modifying originals
    X_train_enhanced = X_train.copy()
    X_test_enhanced = X_test.copy()
    
    # Get common columns
    common_columns = list(set(X_train.columns).intersection(set(X_test.columns)))
    print(f"Common columns: {len(common_columns)}")
    
    # Ensure both datasets have the same columns before transformations
    X_train_enhanced = X_train_enhanced[common_columns]
    X_test_enhanced = X_test_enhanced[common_columns]
    
    # Create log transformations for positive columns
    for col in common_columns:
        if X_train_enhanced[col].min() >= 0 and X_test_enhanced[col].min() >= 0:
            X_train_enhanced[f'{col}_log'] = np.log1p(X_train_enhanced[col])
            X_test_enhanced[f'{col}_log'] = np.log1p(X_test_enhanced[col])
            
            X_train_enhanced[f'{col}_sqrt'] = np.sqrt(X_train_enhanced[col])
            X_test_enhanced[f'{col}_sqrt'] = np.sqrt(X_test_enhanced[col])
        
        # Squared feature (safe for all numeric values)
        X_train_enhanced[f'{col}_squared'] = X_train_enhanced[col] ** 2
        X_test_enhanced[f'{col}_squared'] = X_test_enhanced[col] ** 2
    
    # Create quantile bins using a fitted transformer (ensures consistency)
    numeric_cols = X_train_enhanced.select_dtypes(include=['float64', 'int64']).columns.tolist()
    for col in numeric_cols[:10]:  # Limit to top 10 to avoid excessive feature expansion
        qt = QuantileTransformer(n_quantiles=5, output_distribution='uniform')
        values = qt.fit_transform(X_train_enhanced[col].values.reshape(-1, 1))
        X_train_enhanced[f'{col}_quantile'] = values.flatten()
        
        # Apply same transformation to test
        values_test = qt.transform(X_test_enhanced[col].values.reshape(-1, 1))
        X_test_enhanced[f'{col}_quantile'] = values_test.flatten()
    
    # Create a few key interactions (limit number to avoid feature explosion)
    top_cols = numeric_cols[:5]
    for i, col1 in enumerate(top_cols):
        for col2 in top_cols[i+1:]:
            # Multiply
            X_train_enhanced[f'{col1}_{col2}_mult'] = X_train_enhanced[col1] * X_train_enhanced[col2]
            X_test_enhanced[f'{col1}_{col2}_mult'] = X_test_enhanced[col1] * X_test_enhanced[col2]
            
            # Add
            X_train_enhanced[f'{col1}_{col2}_add'] = X_train_enhanced[col1] + X_train_enhanced[col2]
            X_test_enhanced[f'{col1}_{col2}_add'] = X_test_enhanced[col1] + X_test_enhanced[col2]
            
            # Difference
            X_train_enhanced[f'{col1}_{col2}_diff'] = X_train_enhanced[col1] - X_train_enhanced[col2]
            X_test_enhanced[f'{col1}_{col2}_diff'] = X_test_enhanced[col1] - X_test_enhanced[col2]
    
    # Final verification - ensure both have identical column sets
    train_cols = set(X_train_enhanced.columns)
    test_cols = set(X_test_enhanced.columns)
    
    if train_cols != test_cols:
        diff = train_cols.symmetric_difference(test_cols)
        print(f"Warning: Column mismatch! Differing columns: {diff}")
        # Keep only common columns
        common = list(train_cols.intersection(test_cols))
        X_train_enhanced = X_train_enhanced[common]
        X_test_enhanced = X_test_enhanced[common]
    
    print(f"Original train features: {X_train.shape[1]}")
    print(f"Original test features: {X_test.shape[1]}")
    print(f"Enhanced train features: {X_train_enhanced.shape[1]}")
    print(f"Enhanced test features: {X_test_enhanced.shape[1]}")
    print(f"Verified same feature count: {X_train_enhanced.shape[1] == X_test_enhanced.shape[1]}")
    
    return X_train_enhanced, X_test_enhanced

def train_ensemble_model(X_train, y_train, X_test, test_ids, n_folds=5):
    """Train and ensemble XGBoost and LightGBM models with consistent features"""
    # Set up cross-validation
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize prediction arrays
    oof_preds_xgb = np.zeros(len(X_train))
    oof_preds_lgb = np.zeros(len(X_train))
    test_preds_xgb = np.zeros(len(X_test))
    test_preds_lgb = np.zeros(len(X_test))
    
    # Train across folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"Training fold {fold+1}/{n_folds}")
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # XGBoost model with conservative parameters
        xgb_model = xgb.XGBRegressor(
            n_estimators=2000,
            learning_rate=0.02,
            max_depth=6,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.01,
            reg_lambda=1.0,
            random_state=42+fold,
            objective='binary:logistic',
            eval_metric='auc'
        )
        
        # Train XGBoost
        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=100
        )
        
        # Save model
        pickle.dump(xgb_model, open(f'models/advanced/xgb_fold_{fold}.pkl', 'wb'))
        
        # LightGBM model
        lgb_model = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.02,
            num_leaves=31,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.01,
            reg_lambda=1.0,
            random_state=42+fold,
            objective='binary'
        )
        
        # Train LightGBM
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            early_stopping_rounds=100,
            verbose=100
        )
        
        # Save model
        pickle.dump(lgb_model, open(f'models/advanced/lgb_fold_{fold}.pkl', 'wb'))
        
        # OOF predictions
        oof_preds_xgb[val_idx] = xgb_model.predict(X_val)
        oof_preds_lgb[val_idx] = lgb_model.predict(X_val)
        
        # Test predictions
        test_preds_xgb += xgb_model.predict(X_test) / n_folds
        test_preds_lgb += lgb_model.predict(X_test) / n_folds
        
        # Fold metrics
        xgb_fold_auc = roc_auc_score(y_val, oof_preds_xgb[val_idx])
        lgb_fold_auc = roc_auc_score(y_val, oof_preds_lgb[val_idx])
        
        print(f"Fold {fold+1} - XGB AUC: {xgb_fold_auc:.6f}, LGB AUC: {lgb_fold_auc:.6f}")
        
    # Overall CV metrics
    xgb_cv_auc = roc_auc_score(y_train, oof_preds_xgb)
    lgb_cv_auc = roc_auc_score(y_train, oof_preds_lgb)
    
    print(f"\nFinal CV scores:")
    print(f"XGBoost CV AUC: {xgb_cv_auc:.6f}")
    print(f"LightGBM CV AUC: {lgb_cv_auc:.6f}")
    
    # Create weighted ensemble
    ensemble_weight_xgb = xgb_cv_auc / (xgb_cv_auc + lgb_cv_auc)
    ensemble_weight_lgb = lgb_cv_auc / (xgb_cv_auc + lgb_cv_auc)
    
    ensemble_preds = (test_preds_xgb * ensemble_weight_xgb) + (test_preds_lgb * ensemble_weight_lgb)
    
    print(f"Ensemble weights - XGB: {ensemble_weight_xgb:.4f}, LGB: {ensemble_weight_lgb:.4f}")
    
    # Create submissions
    ensemble_sub = pd.DataFrame({'id': test_ids, 'rainfall': ensemble_preds})
    xgb_sub = pd.DataFrame({'id': test_ids, 'rainfall': test_preds_xgb})
    lgb_sub = pd.DataFrame({'id': test_ids, 'rainfall': test_preds_lgb})
    
    # Save submissions
    ensemble_sub.to_csv('data/submission_ensemble_fixed.csv', index=False)
    xgb_sub.to_csv('data/submission_xgb_fixed.csv', index=False)
    lgb_sub.to_csv('data/submission_lgb_fixed.csv', index=False)
    
    print("\nSubmissions saved to:")
    print("- data/submission_ensemble_fixed.csv")
    print("- data/submission_xgb_fixed.csv")
    print("- data/submission_lgb_fixed.csv")
    
    # Validate with synthetic data if available
    try:
        from validate_submission import validate_submission
        print("\nValidating ensemble submission against synthetic data...")
        ensemble_result = validate_submission('data/submission_ensemble_fixed.csv', quiet=False)
    except Exception as e:
        print(f"Could not validate with synthetic data: {str(e)}")
    
    return {
        'ensemble_sub': ensemble_sub,
        'xgb_sub': xgb_sub,
        'lgb_sub': lgb_sub,
        'xgb_cv_auc': xgb_cv_auc,
        'lgb_cv_auc': lgb_cv_auc
    }

if __name__ == "__main__":
    print("Starting fixed advanced model training...")
    
    # Load data
    X, y, X_test, train_ids, test_ids = load_and_prepare_data()
    
    # Create consistent features
    X_enhanced, X_test_enhanced = create_consistent_features(X, X_test)
    
    # Train models
    results = train_ensemble_model(X_enhanced, y, X_test_enhanced, test_ids)
    
    print("Advanced model training complete!")
