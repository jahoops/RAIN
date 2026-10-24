import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, KFold
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import optuna
from typing import Dict, List, Tuple
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data():
    """Load and prepare data with advanced preprocessing"""
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

def create_advanced_features(X: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create advanced features including interactions and transformations"""
    print("Creating advanced features...")
    
    # Copy dataframes to avoid modifying originals
    X_enhanced = X.copy()
    X_test_enhanced = X_test.copy()
    
    # Function to add transformations for numerical features
    def add_transformations(df):
        transformed_df = df.copy()
        
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            if df[col].min() >= 0:  # Only apply for non-negative values
                # Log transformation (with handling of zeros)
                transformed_df[f'{col}_log'] = np.log1p(df[col])
                
                # Square root transformation
                transformed_df[f'{col}_sqrt'] = np.sqrt(df[col])
            
            # Squared feature (for capturing quadratic relationships)
            transformed_df[f'{col}_squared'] = df[col] ** 2
            
            # Create binned versions (for capturing non-linear patterns)
            transformed_df[f'{col}_bin5'] = pd.qcut(df[col], 5, labels=False, duplicates='drop')
        
        return transformed_df
    
    # Add basic transformations
    X_enhanced = add_transformations(X_enhanced)
    X_test_enhanced = add_transformations(X_test_enhanced)
    
    # Identify top features based on correlation with target (need to run this function with target)
    def get_top_features(X_train, y_train, n=10):
        corr_values = []
        for col in X_train.columns:
            corr = np.corrcoef(X_train[col], y_train)[0, 1]
            corr_values.append((col, abs(corr)))
        
        # Sort by absolute correlation
        corr_values.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in corr_values[:n]]
    
    # This is a placeholder since we don't have y_train in this function
    # In practice, you would pass y_train or use feature importance from a model
    top_features = X_enhanced.columns[:10]  # Using first 10 as placeholder
    
    # Create interaction features between top features
    for i, feat1 in enumerate(top_features):
        if feat1 not in X_enhanced.columns:
            continue
        
        for feat2 in top_features[i+1:]:
            if feat2 not in X_enhanced.columns:
                continue
                
            # Multiply interaction
            X_enhanced[f'{feat1}_{feat2}_mult'] = X_enhanced[feat1] * X_enhanced[feat2]
            X_test_enhanced[f'{feat1}_{feat2}_mult'] = X_test_enhanced[feat1] * X_test_enhanced[feat2]
            
            # Divide interaction (with handling of zeros)
            X_enhanced[f'{feat1}_{feat2}_div'] = X_enhanced[feat1] / (X_enhanced[feat2] + 1e-5)
            X_test_enhanced[f'{feat1}_{feat2}_div'] = X_test_enhanced[feat1] / (X_test_enhanced[feat2] + 1e-5)
            
            # Addition and subtraction
            X_enhanced[f'{feat1}_{feat2}_add'] = X_enhanced[feat1] + X_enhanced[feat2]
            X_test_enhanced[f'{feat1}_{feat2}_add'] = X_test_enhanced[feat1] + X_test_enhanced[feat2]
            X_enhanced[f'{feat1}_{feat2}_sub'] = X_enhanced[feat1] - X_enhanced[feat2]
            X_test_enhanced[f'{feat1}_{feat2}_sub'] = X_test_enhanced[feat1] - X_test_enhanced[feat2]
    
    # Print feature counts
    print(f"Original feature count: {X.shape[1]}")
    print(f"Enhanced feature count: {X_enhanced.shape[1]}")
    
    return X_enhanced, X_test_enhanced

def optimize_xgboost(trial, X, y, cv_folds=5):
    """Optimize XGBoost hyperparameters using Optuna"""
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': trial.suggest_categorical('booster', ['gbtree']),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'eta': trial.suggest_float('eta', 0.01, 0.2, log=True),
        'gamma': trial.suggest_float('gamma', 1e-3, 1.0, log=True),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
    }
    
    # Use stratified K-fold cross-validation
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train model
        model = xgb.train(
            param, 
            dtrain, 
            num_boost_round=1000, 
            evals=[(dval, 'validation')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Predict and evaluate
        preds = model.predict(dval)
        auc = roc_auc_score(y_val, preds)
        scores.append(auc)
    
    # Return the mean cross-validation AUC
    return np.mean(scores)

def optimize_lightgbm(trial, X, y, cv_folds=5):
    """Optimize LightGBM hyperparameters using Optuna"""
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
    }
    
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val)
        
        # Train model
        model = lgb.train(
            param, 
            dtrain, 
            num_boost_round=1000, 
            valid_sets=[dval], 
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        # Predict and evaluate
        preds = model.predict(X_val)
        auc = roc_auc_score(y_val, preds)
        scores.append(auc)
    
    return np.mean(scores)

def train_optimized_models(X, y, X_test, n_trials=500, cv_folds=5):
    """Train optimized models using hyperparameter tuning"""
    print("Starting hyperparameter optimization...")
    
    # Create a study for XGBoost
    xgb_study = optuna.create_study(direction='maximize')
    xgb_study.optimize(lambda trial: optimize_xgboost(trial, X, y, cv_folds), n_trials=n_trials)
    
    # Create a study for LightGBM
    lgb_study = optuna.create_study(direction='maximize')
    lgb_study.optimize(lambda trial: optimize_lightgbm(trial, X, y, cv_folds), n_trials=n_trials)
    
    print(f"Best XGBoost score: {xgb_study.best_value:.6f}")
    print(f"Best LightGBM score: {lgb_study.best_value:.6f}")
    
    # Save best parameters
    best_params = {
        'xgboost': xgb_study.best_params,
        'lightgbm': lgb_study.best_params
    }
    
    with open('models/advanced/best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)
    
    print("Best parameters saved to 'models/advanced/best_params.pkl'")
    
    # Train final models with best parameters
    return train_final_models(X, y, X_test, best_params)

def train_final_models(X, y, X_test, best_params=None, n_folds=5):
    """Train final models using the best hyperparameters"""
    print("Training final models...")
    
    # If no best parameters provided, try to load them
    if best_params is None:
        try:
            with open('models/advanced/best_params.pkl', 'rb') as f:
                best_params = pickle.load(f)
        except:
            print("No best parameters found. Using defaults.")
            best_params = {
                'xgboost': {
                    'booster': 'gbtree', 
                    'lambda': 1.0,
                    'alpha': 0.01,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'max_depth': 6,
                    'eta': 0.03,
                    'gamma': 0.01,
                    'grow_policy': 'depthwise',
                    'min_child_weight': 1
                },
                'lightgbm': {
                    'lambda_l1': 0.01,
                    'lambda_l2': 0.01,
                    'num_leaves': 31,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 1,
                    'min_child_samples': 20,
                    'learning_rate': 0.03
                }
            }
    
    # Set up cross-validation
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize arrays for predictions
    xgb_oof_preds = np.zeros(X.shape[0])
    lgb_oof_preds = np.zeros(X.shape[0])
    xgb_test_preds = np.zeros(X_test.shape[0])
    lgb_test_preds = np.zeros(X_test.shape[0])
    
    # Store fold models
    xgb_models = []
    lgb_models = []
    
    # Variable to track feature importance across folds
    xgb_importance = pd.DataFrame()
    lgb_importance = pd.DataFrame()
    
    # Train models using cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"Training fold {fold+1}/{n_folds}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train XGBoost model
        xgb_model = xgb.XGBRegressor(
            **best_params['xgboost'],
            objective='binary:logistic',
            eval_metric='auc',
            n_estimators=2000,
            random_state=42+fold
        )
        
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=100
        )
        
        # Save feature importance for XGBoost
        fold_importance = pd.DataFrame()
        fold_importance['feature'] = X.columns
        fold_importance['importance'] = xgb_model.feature_importances_
        fold_importance['fold'] = fold
        xgb_importance = pd.concat([xgb_importance, fold_importance], axis=0)
        
        # Train LightGBM model
        lgb_model = lgb.LGBMRegressor(
            **best_params['lightgbm'],
            objective='binary',
            n_estimators=2000,
            random_state=42+fold
        )
        
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=100
        )
        
        # Save feature importance for LightGBM
        fold_importance = pd.DataFrame()
        fold_importance['feature'] = X.columns
        fold_importance['importance'] = lgb_model.feature_importances_
        fold_importance['fold'] = fold
        lgb_importance = pd.concat([lgb_importance, fold_importance], axis=0)
        
        # Make predictions for this fold
        xgb_oof_preds[val_idx] = xgb_model.predict(X_val)
        lgb_oof_preds[val_idx] = lgb_model.predict(X_val)
        
        # Make test predictions and accumulate
        xgb_test_preds += xgb_model.predict(X_test) / n_folds
        lgb_test_preds += lgb_model.predict(X_test) / n_folds
        
        # Save models
        xgb_models.append(xgb_model)
        lgb_models.append(lgb_model)
        
        # Save fold models
        pickle.dump(xgb_model, open(f'models/advanced/xgb_fold_{fold}.pkl', 'wb'))
        pickle.dump(lgb_model, open(f'models/advanced/lgb_fold_{fold}.pkl', 'wb'))
        
        # Calculate fold scores
        xgb_fold_auc = roc_auc_score(y_val, xgb_oof_preds[val_idx])
        lgb_fold_auc = roc_auc_score(y_val, lgb_oof_preds[val_idx])
        print(f"Fold {fold+1} - XGB AUC: {xgb_fold_auc:.6f}, LGB AUC: {lgb_fold_auc:.6f}")
    
    # Calculate overall CV scores
    xgb_cv_auc = roc_auc_score(y, xgb_oof_preds)
    lgb_cv_auc = roc_auc_score(y, lgb_oof_preds)
    
    print(f"\nXGBoost CV AUC: {xgb_cv_auc:.6f}")
    print(f"LightGBM CV AUC: {lgb_cv_auc:.6f}")
    
    # Save OOF predictions
    oof_df = pd.DataFrame({
        'xgb_pred': xgb_oof_preds,
        'lgb_pred': lgb_oof_preds,
        'target': y
    })
    oof_df.to_csv('models/advanced/oof_predictions.csv', index=False)
    
    # Save feature importance
    xgb_importance.to_csv('models/advanced/xgb_importance.csv', index=False)
    lgb_importance.to_csv('models/advanced/lgb_importance.csv', index=False)
    
    # Plot feature importance
    plt.figure(figsize=(14, 10))
    
    # XGBoost feature importance
    plt.subplot(1, 2, 1)
    xgb_imp = xgb_importance.groupby('feature')['importance'].mean().sort_values(ascending=False).head(20)
    sns.barplot(x=xgb_imp.values, y=xgb_imp.index)
    plt.title('XGBoost Feature Importance')
    
    # LightGBM feature importance
    plt.subplot(1, 2, 2)
    lgb_imp = lgb_importance.groupby('feature')['importance'].mean().sort_values(ascending=False).head(20)
    sns.barplot(x=lgb_imp.values, y=lgb_imp.index)
    plt.title('LightGBM Feature Importance')
    
    plt.tight_layout()
    plt.savefig('models/advanced/feature_importance.png')
    
    return {
        'xgb_models': xgb_models,
        'lgb_models': lgb_models,
        'xgb_preds': xgb_test_preds,
        'lgb_preds': lgb_test_preds,
        'xgb_cv_auc': xgb_cv_auc,
        'lgb_cv_auc': lgb_cv_auc
    }

def create_ensemble_submission(model_results, test_ids):
    """Create a weighted ensemble of all model predictions"""
    print("Creating ensemble submission...")
    
    # Get predictions from different models
    xgb_preds = model_results['xgb_preds']
    lgb_preds = model_results['lgb_preds']
    
    # Get CV scores for weighting
    xgb_cv = model_results['xgb_cv_auc']
    lgb_cv = model_results['lgb_cv_auc']
    
    # Calculate weights based on CV performance
    total_cv = xgb_cv + lgb_cv
    xgb_weight = xgb_cv / total_cv
    lgb_weight = lgb_cv / total_cv
    
    print(f"Ensemble weights - XGB: {xgb_weight:.4f}, LGB: {lgb_weight:.4f}")
    
    # Create weighted ensemble predictions
    ensemble_preds = (xgb_preds * xgb_weight) + (lgb_preds * lgb_weight)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'id': test_ids,
        'rainfall': ensemble_preds
    })
    
    # Save individual model predictions
    xgb_submission = pd.DataFrame({'id': test_ids, 'rainfall': xgb_preds})
    lgb_submission = pd.DataFrame({'id': test_ids, 'rainfall': lgb_preds})
    
    # Save all submissions
    submission.to_csv('data/submission_ensemble.csv', index=False)
    xgb_submission.to_csv('data/submission_xgb.csv', index=False)
    lgb_submission.to_csv('data/submission_lgb.csv', index=False)
    
    print("Saved submissions:")
    print("- data/submission_ensemble.csv")
    print("- data/submission_xgb.csv")
    print("- data/submission_lgb.csv")
    
    # Validate with synthetic data if available
    try:
        from validate_submission import validate_submission
        print("\nValidating ensemble submission against synthetic data...")
        ensemble_result = validate_submission('data/submission_ensemble.csv', quiet=True)
        
        print("Synthetic validation results:")
        print(f"Estimated contest score: {ensemble_result['estimated_score']:.6f}")
        if ensemble_result['estimated_score'] >= 0.85:
            print("EXCELLENT - On par with or exceeding benchmark!")
        elif ensemble_result['estimated_score'] >= 0.83:
            print("VERY GOOD - Close to benchmark performance")
    except Exception as e:
        print(f"Could not validate with synthetic data: {str(e)}")
    
    return submission

def train_advanced_model():
    """Train advanced ensemble model with all optimizations"""
    # Load and prepare data
    X, y, X_test, train_ids, test_ids = load_and_prepare_data()
    
    # Create advanced features
    X_enhanced, X_test_enhanced = create_advanced_features(X, X_test)
    
    # Optimize and train models
    model_results = train_optimized_models(X_enhanced, y, X_test_enhanced, n_trials=500, cv_folds=5)
    
    # Create ensemble submission
    submission = create_ensemble_submission(model_results, test_ids)
    
    return submission

if __name__ == "__main__":
    print("Starting advanced model training pipeline...")
    submission = train_advanced_model()
    print("Advanced model training complete!")
