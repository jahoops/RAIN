import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

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
    train_ids = train_df['id'].copy()
    test_ids = test_df['id'].copy()
    
    # Extract target
    y = train_df['rainfall'].copy()
    
    # Drop non-feature columns
    X = train_df.drop(['id', 'rainfall'], axis=1).copy()
    X_test = test_df.drop(['id'], axis=1).copy()
    
    # Create directory for models if not exists
    os.makedirs('models/fixed', exist_ok=True)
    
    return X, y, X_test, train_ids, test_ids

def add_safe_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create features guaranteed to be consistent between train and test"""
    print("Creating consistent features...")
    
    # Step 1: Ensure both dataframes have exactly the same columns in the same order
    common_columns = [col for col in X_train.columns if col in X_test.columns]
    print(f"Common columns: {len(common_columns)} out of {len(X_train.columns)}")
    
    X_train_filtered = X_train[common_columns].copy()
    X_test_filtered = X_test[common_columns].copy()
    
    # Step 2: Create a combined DataFrame for consistent transformations
    n_train = len(X_train_filtered)
    combined = pd.concat([X_train_filtered, X_test_filtered], axis=0)
    print(f"Combined shape: {combined.shape}")
    
    # Step 3: Safe transformations that guarantee consistency
    for col in combined.select_dtypes(include=['float64', 'int64']).columns:
        # Log transform for non-negative columns
        if combined[col].min() >= 0:
            combined[f'{col}_log'] = np.log1p(combined[col])
            combined[f'{col}_sqrt'] = np.sqrt(combined[col])
        
        # Square features (always safe)
        combined[f'{col}_squared'] = combined[col] ** 2
        
        # Create fixed bins using quantile ranges determined on the combined data
        # This ensures the same bins are used for both train and test
        try:
            bins = np.linspace(0, 1, 6)  # 5 bins
            quantiles = combined[col].quantile(bins).values
            # Ensure unique values to avoid warnings
            quantiles = np.unique(quantiles)
            
            # Only proceed if we have enough unique quantiles
            if len(quantiles) >= 3:
                bin_labels = list(range(len(quantiles)-1))
                combined[f'{col}_bin'] = pd.cut(
                    combined[col], 
                    bins=quantiles, 
                    labels=bin_labels,
                    include_lowest=True
                ).astype('float32')  # Convert to float to handle missing values
        except Exception as e:
            print(f"Skipping binning for {col}: {str(e)}")
    
    # Step 4: Create simple feature interactions for top N features
    numeric_cols = combined.select_dtypes(['float64', 'int64']).columns.tolist()[:10]
    
    for i, col1 in enumerate(numeric_cols[:5]):  # Limit to first 5 features
        for col2 in numeric_cols[i+1:5]:  # And their interactions
            # Safe interactions
            combined[f'{col1}_{col2}_sum'] = combined[col1] + combined[col2]
            combined[f'{col1}_{col2}_diff'] = combined[col1] - combined[col2]
            combined[f'{col1}_{col2}_mult'] = combined[col1] * combined[col2]
    
    # Step 5: Split back the combined DataFrame
    X_train_enhanced = combined.iloc[:n_train].copy()
    X_test_enhanced = combined.iloc[n_train:].copy()
    
    # Final verification - columns MUST match exactly in name and order
    assert X_train_enhanced.columns.equals(X_test_enhanced.columns), "Column mismatch detected!"
    assert X_train_enhanced.shape[1] == X_test_enhanced.shape[1], "Feature count mismatch!"
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Enhanced features: {X_train_enhanced.shape[1]}")
    print(f"Feature consistency verified ✓")
    
    return X_train_enhanced, X_test_enhanced

def train_model(X_train, y_train, X_test, test_ids, n_folds=5):
    """Train robust model with consistent features"""
    print("\nTraining model with cross-validation...")
    
    # Create consistent train/validation splits
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize arrays for predictions
    xgb_oof = np.zeros(len(X_train))
    lgb_oof = np.zeros(len(X_train))
    xgb_preds = np.zeros(len(X_test))
    lgb_preds = np.zeros(len(X_test))
    
    # Track scores across folds
    fold_scores_xgb = []
    fold_scores_lgb = []
    
    # Train across folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        
        # Split data
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # XGBoost with conservative hyperparameters
        print("Training XGBoost model...")
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 5,
            'learning_rate': 0.02,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.01,
            'reg_lambda': 1,
            'min_child_weight': 3
        }
        
        xgb_model = xgb.XGBRegressor(
            **xgb_params,
            n_estimators=2000,
            random_state=42+fold
        )
        
        # Train with early stopping
        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=50
        )
        
        # Save model
        pickle.dump(xgb_model, open(f'models/fixed/xgb_fold_{fold}.pkl', 'wb'))
        
        # LightGBM model
        print("Training LightGBM model...")
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.02,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'verbose': -1
        }
        
        lgb_model = lgb.LGBMRegressor(
            **lgb_params,
            n_estimators=2000,
            random_state=42+fold
        )
        
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            early_stopping_rounds=100,
            verbose=50
        )
        
        # Save model
        pickle.dump(lgb_model, open(f'models/fixed/lgb_fold_{fold}.pkl', 'wb'))
        
        # Make out-of-fold predictions
        xgb_val_preds = xgb_model.predict(X_val)
        lgb_val_preds = lgb_model.predict(X_val)
        
        xgb_oof[val_idx] = xgb_val_preds
        lgb_oof[val_idx] = lgb_val_preds
        
        # Make test predictions
        xgb_preds += xgb_model.predict(X_test) / n_folds
        lgb_preds += lgb_model.predict(X_test) / n_folds
        
        # Calculate fold scores
        xgb_fold_score = roc_auc_score(y_val, xgb_val_preds)
        lgb_fold_score = roc_auc_score(y_val, lgb_val_preds)
        
        fold_scores_xgb.append(xgb_fold_score)
        fold_scores_lgb.append(lgb_fold_score)
        
        print(f"Fold {fold+1} - XGB AUC: {xgb_fold_score:.6f}, LGB AUC: {lgb_fold_score:.6f}")
    
    # Calculate overall CV scores
    xgb_cv_score = roc_auc_score(y_train, xgb_oof)
    lgb_cv_score = roc_auc_score(y_train, lgb_oof)
    
    print(f"\nXGBoost CV Score: {xgb_cv_score:.6f}")
    print(f"LightGBM CV Score: {lgb_cv_score:.6f}")
    
    # Create ensemble with weighted average based on CV performance
    weight_xgb = xgb_cv_score / (xgb_cv_score + lgb_cv_score)
    weight_lgb = lgb_cv_score / (xgb_cv_score + lgb_cv_score)
    
    print(f"Ensemble weights - XGB: {weight_xgb:.4f}, LGB: {weight_lgb:.4f}")
    
    ensemble_preds = (xgb_preds * weight_xgb) + (lgb_preds * weight_lgb)
    
    # Create submissions
    ensemble_submission = pd.DataFrame({'id': test_ids, 'rainfall': ensemble_preds})
    xgb_submission = pd.DataFrame({'id': test_ids, 'rainfall': xgb_preds})
    lgb_submission = pd.DataFrame({'id': test_ids, 'rainfall': lgb_preds})
    
    # Save to file
    ensemble_submission.to_csv('data/submission_ensemble_fixed2.csv', index=False)
    xgb_submission.to_csv('data/submission_xgb_fixed2.csv', index=False)
    lgb_submission.to_csv('data/submission_lgb_fixed2.csv', index=False)
    
    print("\nSaved submissions:")
    print("- data/submission_ensemble_fixed2.csv")
    print("- data/submission_xgb_fixed2.csv")
    print("- data/submission_lgb_fixed2.csv")
    
    # Create submissions folder and copy files there for safety
    os.makedirs('models/fixed/submissions', exist_ok=True)
    for file in ['submission_ensemble_fixed2.csv', 'submission_xgb_fixed2.csv', 'submission_lgb_fixed2.csv']:
        src = os.path.join('data', file)
        dst = os.path.join('models/fixed/submissions', file)
        shutil.copy2(src, dst)
    
    return {
        'xgb_cv_score': xgb_cv_score,
        'lgb_cv_score': lgb_cv_score,
        'ensemble_preds': ensemble_preds,
        'xgb_preds': xgb_preds,
        'lgb_preds': lgb_preds
    }

def validate_with_synthetic(results):
    """Validate results using synthetic data"""
    print("\nValidating with synthetic data...")
    
    try:
        from validate_submission import validate_submission
        
        # Try to validate all submissions
        for file in ['submission_ensemble_fixed2.csv', 'submission_xgb_fixed2.csv', 'submission_lgb_fixed2.csv']:
            print(f"\nValidating {file}...")
            result = validate_submission(f'data/{file}', quiet=False)
            
            # Copy to keep directory if validation is successful
            if result and result.get('estimated_score', 0) > 0.8:
                os.makedirs('data/keep', exist_ok=True)
                src = os.path.join('data', file)
                dst = os.path.join('data/keep', file)
                shutil.copy2(src, dst)
                print(f"✓ Saved {file} to trusted keep directory")
    
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        print("Please run validation manually with validate_submission.py")

def run_model():
    """Main function to run the fixed modeling pipeline"""
    print("\n===== RAIN MODEL (GUARANTEED FEATURE CONSISTENCY) =====")
    
    # Step 1: Load and prepare data
    print("\nStep 1: Loading and preprocessing data...")
    X, y, X_test, train_ids, test_ids = load_and_prepare_data()
    
    # Step 2: Create consistent features
    print("\nStep 2: Creating consistent features...")
    X_enhanced, X_test_enhanced = add_safe_features(X, X_test)
    
    # Step 3: Train model
    print("\nStep 3: Training model...")
    results = train_model(X_enhanced, y, X_test_enhanced, test_ids)
    
    # Step 4: Validate with synthetic data
    print("\nStep 4: Running validation...")
    validate_with_synthetic(results)
    
    print("\n===== MODELING COMPLETE =====")
    return results

if __name__ == "__main__":
    run_model()
