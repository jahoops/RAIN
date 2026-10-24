import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import pickle
import os

def fix_feature_mismatch():
    """Fix the feature mismatch in the advanced model"""
    print("\n===== FIXING FEATURE MISMATCH IN ADVANCED MODEL =====")
    
    # 1. First load the data 
    from data_loader import load_data, preprocess_data
    from feature_engineering import engineer_features
    
    print("1. Loading and preprocessing data...")
    train_df, test_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df)
    train_df, test_df = engineer_features(train_df, test_df, strategy='all')
    
    # 2. Extract features and target
    train_ids = train_df['id']
    test_ids = test_df['id']
    y = train_df['rainfall']
    X = train_df.drop(['id', 'rainfall'], axis=1)
    X_test = test_df.drop(['id'], axis=1)
    
    print(f"Original train shape: {X.shape}")
    print(f"Original test shape: {X_test.shape}")
    
    # 3. Fix: Ensure feature consistency between train and test
    common_columns = list(set(X.columns).intersection(set(X_test.columns)))
    print(f"Common columns: {len(common_columns)}")
    
    # Use only common features
    X = X[common_columns]
    X_test = X_test[common_columns]
    
    print(f"After alignment - Train shape: {X.shape}, Test shape: {X_test.shape}")
    
    # 4. Fix: Create a combined dataset approach for feature engineering
    n_train = X.shape[0]
    combined = pd.concat([X, X_test], axis=0).reset_index(drop=True)
    
    # 5. Now we perform feature engineering on the combined dataset
    print("\n2. Creating consistent features...")
    
    # Apply transformations to combined data
    for col in combined.select_dtypes(include=['float64', 'int64']).columns:
        # Log transform for non-negative columns
        if combined[col].min() >= 0:
            combined[f'{col}_log'] = np.log1p(combined[col])
            combined[f'{col}_sqrt'] = np.sqrt(combined[col])
        
        # Square features
        combined[f'{col}_squared'] = combined[col] ** 2
    
    # Create interaction features
    top_features = combined.columns[:5]  # Use first 5 features for interactions
    for i, feat1 in enumerate(top_features):
        for feat2 in top_features[i+1:]:
            # Multiply interaction
            combined[f'{feat1}_{feat2}_mult'] = combined[feat1] * combined[feat2]
            
            # Add/subtract
            combined[f'{feat1}_{feat2}_add'] = combined[feat1] + combined[feat2]
            combined[f'{feat1}_{feat2}_diff'] = combined[feat1] - combined[feat2]
    
    # Split back into train and test
    X_enhanced = combined.iloc[:n_train]
    X_test_enhanced = combined.iloc[n_train:]
    
    # Verify consistency
    print(f"Enhanced - Train shape: {X_enhanced.shape}, Test shape: {X_test_enhanced.shape}")
    assert X_enhanced.shape[1] == X_test_enhanced.shape[1], "Feature count mismatch!"
    
    # 6. Train a simple model with these consistent features
    print("\n3. Training fixed model...")
    
    # Set up cross-validation
    n_folds = 5
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize arrays for predictions
    xgb_oof = np.zeros(len(X_enhanced))
    xgb_preds = np.zeros(len(X_test_enhanced))
    lgb_oof = np.zeros(len(X_enhanced))
    lgb_preds = np.zeros(len(X_test_enhanced))
    
    # Create directory for models
    os.makedirs('models/fixed', exist_ok=True)
    
    # Train models across folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_enhanced, y)):
        print(f"Training fold {fold+1}/{n_folds}")
        X_tr, X_val = X_enhanced.iloc[train_idx], X_enhanced.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # XGBoost model
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.01,
            'reg_lambda': 1.0,
            'min_child_weight': 3,
            'random_state': 42 + fold
        }
        
        xgb_model = xgb.XGBRegressor(**xgb_params, n_estimators=1000)
        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # LightGBM model
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 31,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'random_state': 42 + fold,
            'verbose': -1
        }
        
        lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=1000)
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Save models
        pickle.dump(xgb_model, open(f'models/fixed/xgb_fold_{fold}.pkl', 'wb'))
        pickle.dump(lgb_model, open(f'models/fixed/lgb_fold_{fold}.pkl', 'wb'))
        
        # Out-of-fold predictions
        xgb_oof[val_idx] = xgb_model.predict(X_val)
        lgb_oof[val_idx] = lgb_model.predict(X_val)
        
        # Test predictions
        xgb_preds += xgb_model.predict(X_test_enhanced) / n_folds
        lgb_preds += lgb_model.predict(X_test_enhanced) / n_folds
        
        # Calculate fold AUC
        xgb_fold_auc = roc_auc_score(y_val, xgb_oof[val_idx])
        lgb_fold_auc = roc_auc_score(y_val, lgb_oof[val_idx])
        print(f"Fold {fold+1} - XGB AUC: {xgb_fold_auc:.6f}, LGB AUC: {lgb_fold_auc:.6f}")
    
    # Overall CV AUC
    xgb_auc = roc_auc_score(y, xgb_oof)
    lgb_auc = roc_auc_score(y, lgb_oof)
    
    print(f"\nFinal CV scores:")
    print(f"XGBoost CV AUC: {xgb_auc:.6f}")
    print(f"LightGBM CV AUC: {lgb_auc:.6f}")
    
    # Create weighted ensemble
    weight_xgb = xgb_auc / (xgb_auc + lgb_auc)
    weight_lgb = lgb_auc / (xgb_auc + lgb_auc)
    
    ensemble_preds = (xgb_preds * weight_xgb) + (lgb_preds * weight_lgb)
    
    # Create submissions
    ensemble_sub = pd.DataFrame({'id': test_ids, 'rainfall': ensemble_preds})
    xgb_sub = pd.DataFrame({'id': test_ids, 'rainfall': xgb_preds})
    lgb_sub = pd.DataFrame({'id': test_ids, 'rainfall': lgb_preds})
    
    # Save submissions
    ensemble_sub.to_csv('data/submission_fixed_ensemble.csv', index=False)
    xgb_sub.to_csv('data/submission_fixed_xgb.csv', index=False)
    lgb_sub.to_csv('data/submission_fixed_lgb.csv', index=False)
    
    print("\nSaved submissions:")
    print("- data/submission_fixed_ensemble.csv")
    print("- data/submission_fixed_xgb.csv")
    print("- data/submission_fixed_lgb.csv")
    
    # Try validation if available
    try:
        from validate_submission import validate_submission
        print("\nValidating ensemble submission:")
        validate_submission('data/submission_fixed_ensemble.csv')
    except Exception as e:
        print(f"Validation not available: {e}")
    
    print("\nFeature mismatch issue fixed successfully!")

if __name__ == "__main__":
    fix_feature_mismatch()
