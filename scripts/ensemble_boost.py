import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple
import pickle
import shutil
from run_fixed_model import add_safe_features

def load_data():
    """Load train and test data with consistent feature processing"""
    from data_loader import load_data, preprocess_data
    from feature_engineering import engineer_features
    
    # Load base data
    train_df, test_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # Apply feature engineering
    train_df, test_df = engineer_features(train_df, test_df, strategy='all')
    
    # Extract components
    train_ids = train_df['id']
    test_ids = test_df['id']
    y = train_df['rainfall']
    
    # Extract features
    X = train_df.drop(['id', 'rainfall'], axis=1)
    X_test = test_df.drop(['id'], axis=1)
    
    # Ensure feature consistency
    common_cols = list(set(X.columns).intersection(set(X_test.columns)))
    X = X[common_cols]
    X_test = X_test[common_cols]
    
    # Create advanced features
    X, X_test = add_safe_features(X, X_test)
    
    return X, y, X_test, train_ids, test_ids

def train_three_way_ensemble(n_folds=5):
    """
    Train an improved three-way ensemble using XGBoost, LightGBM, and CatBoost
    with correlation-based model weighting
    """
    print("\n===== TRAINING IMPROVED THREE-WAY ENSEMBLE =====")
    
    # Set up output directory
    os.makedirs('models/ensemble', exist_ok=True)
    
    # Load data
    print("\nLoading and preparing data...")
    X, y, X_test, train_ids, test_ids = load_data()
    
    print(f"Train shape: {X.shape}")
    print(f"Test shape: {X_test.shape}")
    
    # Set up cross-validation
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize arrays for out-of-fold and test predictions
    oof_preds_xgb = np.zeros(len(X))
    oof_preds_lgb = np.zeros(len(X))
    oof_preds_ctb = np.zeros(len(X))
    test_preds_xgb = np.zeros(len(X_test))
    test_preds_lgb = np.zeros(len(X_test))
    test_preds_ctb = np.zeros(len(X_test))
    
    # Train models with cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n--- Training fold {fold+1}/{n_folds} ---")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 1. Train XGBoost
        print("Training XGBoost...")
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.02,  # Lower learning rate for better generalization
            'subsample': 0.75,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': 42 + fold
        }
        
        xgb_model = xgb.XGBRegressor(
            **xgb_params,
            n_estimators=2000
        )
        
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=50
        )
        
        # Save model
        pickle.dump(xgb_model, open(f'models/ensemble/xgb_fold_{fold}.pkl', 'wb'))
        
        # 2. Train LightGBM
        print("Training LightGBM...")
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 40,
            'learning_rate': 0.02, 
            'feature_fraction': 0.75,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 30,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': 42 + fold,
            'verbosity': -1
        }
        
        lgb_model = lgb.LGBMRegressor(
            **lgb_params,
            n_estimators=2000
        )
        
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=50
        )
        
        # Save model
        pickle.dump(lgb_model, open(f'models/ensemble/lgb_fold_{fold}.pkl', 'wb'))
        
        # 3. Train CatBoost
        print("Training CatBoost...")
        ctb_params = {
            'objective': 'RMSE',  # Using RMSE as CatBoost doesn't have AUC for regression
            'eval_metric': 'RMSE',
            'depth': 7,
            'learning_rate': 0.02,
            'random_seed': 42 + fold,
            'l2_leaf_reg': 3,
            'subsample': 0.8,
            'verbose': 0,
            'bootstrap_type': 'Bernoulli'
        }
        
        ctb_model = ctb.CatBoostRegressor(
            **ctb_params,
            iterations=1500
        )
        
        ctb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=50
        )
        
        # Save model
        ctb_model.save_model(f'models/ensemble/ctb_fold_{fold}.cbm')
        
        # Get out-of-fold predictions
        oof_preds_xgb[val_idx] = xgb_model.predict(X_val)
        oof_preds_lgb[val_idx] = lgb_model.predict(X_val)
        oof_preds_ctb[val_idx] = ctb_model.predict(X_val)
        
        # Get test predictions (average across folds)
        test_preds_xgb += xgb_model.predict(X_test) / n_folds
        test_preds_lgb += lgb_model.predict(X_test) / n_folds
        test_preds_ctb += ctb_model.predict(X_test) / n_folds
        
        # Calculate fold performance
        xgb_fold_score = roc_auc_score(y_val, oof_preds_xgb[val_idx])
        lgb_fold_score = roc_auc_score(y_val, oof_preds_lgb[val_idx])
        ctb_fold_score = roc_auc_score(y_val, oof_preds_ctb[val_idx])
        
        print(f"Fold {fold+1} scores - XGB: {xgb_fold_score:.6f}, LGB: {lgb_fold_score:.6f}, CTB: {ctb_fold_score:.6f}")
    
    # Calculate overall CV scores
    xgb_score = roc_auc_score(y, oof_preds_xgb)
    lgb_score = roc_auc_score(y, oof_preds_lgb)
    ctb_score = roc_auc_score(y, oof_preds_ctb)
    
    print(f"\nFinal CV scores:")
    print(f"XGBoost CV AUC: {xgb_score:.6f}")
    print(f"LightGBM CV AUC: {lgb_score:.6f}") 
    print(f"CatBoost CV AUC: {ctb_score:.6f}")
    
    # Save OOF predictions
    oof_df = pd.DataFrame({
        'id': train_ids,
        'xgb_pred': oof_preds_xgb,
        'lgb_pred': oof_preds_lgb,
        'ctb_pred': oof_preds_ctb,
        'target': y
    })
    oof_df.to_csv('models/ensemble/oof_preds.csv', index=False)
    print("Saved out-of-fold predictions to models/ensemble/oof_preds.csv")
    
    # Correlation-based model blending
    # Calculate correlation of each model's predictions with the target
    corr_xgb = np.corrcoef(y, oof_preds_xgb)[0, 1]
    corr_lgb = np.corrcoef(y, oof_preds_lgb)[0, 1]
    corr_ctb = np.corrcoef(y, oof_preds_ctb)[0, 1]
    
    # Use squared correlations for weighting (emphasizes better models)
    corr_sum = corr_xgb**2 + corr_lgb**2 + corr_ctb**2
    weight_xgb = (corr_xgb**2) / corr_sum
    weight_lgb = (corr_lgb**2) / corr_sum
    weight_ctb = (corr_ctb**2) / corr_sum
    
    print(f"\nEnsemble weights based on squared correlations:")
    print(f"XGBoost: {weight_xgb:.4f}")
    print(f"LightGBM: {weight_lgb:.4f}")
    print(f"CatBoost: {weight_ctb:.4f}")
    
    # Generate blended predictions
    ensemble_preds = (
        test_preds_xgb * weight_xgb + 
        test_preds_lgb * weight_lgb + 
        test_preds_ctb * weight_ctb
    )
    
    # Create submissions
    ensemble_sub = pd.DataFrame({'id': test_ids, 'rainfall': ensemble_preds})
    xgb_sub = pd.DataFrame({'id': test_ids, 'rainfall': test_preds_xgb})
    lgb_sub = pd.DataFrame({'id': test_ids, 'rainfall': test_preds_lgb})
    ctb_sub = pd.DataFrame({'id': test_ids, 'rainfall': test_preds_ctb})
    
    # Save submissions
    ensemble_sub.to_csv('data/submission_ensemble_boost.csv', index=False)
    xgb_sub.to_csv('data/submission_xgb_boost.csv', index=False)
    lgb_sub.to_csv('data/submission_lgb_boost.csv', index=False)
    ctb_sub.to_csv('data/submission_ctb_boost.csv', index=False)
    
    print("\nSaved submissions:")
    print("- data/submission_ensemble_boost.csv")
    print("- data/submission_xgb_boost.csv")
    print("- data/submission_lgb_boost.csv")
    print("- data/submission_ctb_boost.csv")
    
    # Validate with synthetic data
    try:
        print("\nValidating with synthetic data...")
        from validate_submission import validate_submission
        validate_submission('data/submission_ensemble_boost.csv')
        
        # Make a backup if validation score is good
        os.makedirs('data/keep', exist_ok=True)
        shutil.copy2('data/submission_ensemble_boost.csv', 'data/keep/')
        print("Copied ensemble submission to data/keep/ directory for safekeeping.")
    except Exception as e:
        print(f"Error validating submission: {str(e)}")

if __name__ == "__main__":
    train_three_way_ensemble()
