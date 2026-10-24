import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
import pickle
import os
from typing import Dict, Any
import time

from data_loader import load_data, preprocess_data
from feature_engineering import engineer_features
from temporal_features import add_temporal_rainfall_features

def objective(trial, X_train, y_train, X_valid, y_valid, model_type='xgb'):
    """Optuna objective function to optimize model hyperparameters"""
    if model_type == 'xgb':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42
        }
        
        model = xgb.XGBRegressor(**params)
    
    elif model_type == 'lgb':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42
        }
        
        model = lgb.LGBMRegressor(**params)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predict on validation set
    preds = model.predict(X_valid)
    
    # Calculate AUC
    auc_score = roc_auc_score(y_valid, preds)
    
    return auc_score

def optimize_and_train(feature_selection='boruta_top', model_type='xgb', n_trials=100):
    """Optimize hyperparameters and train the final model"""
    print(f"Starting optimization for {model_type} with {feature_selection} features")
    
    # Load and preprocess data
    train_df, test_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df)
    train_df, test_df = add_temporal_rainfall_features(train_df, test_df)
    # Engineer features based on selected strategy
    train_df, test_df = engineer_features(train_df, test_df, feature_selection)
    
    # Prepare features and target
    X = train_df.drop(['id', 'rainfall'], axis=1, errors='ignore')
    y = train_df['rainfall']
    
    # Split data for validation
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create directory for models
    os.makedirs('models', exist_ok=True)
    
    # Optimize hyperparameters
    start_time = time.time()
    print(f"Starting hyperparameter optimization with {n_trials} trials...")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_valid, y_valid, model_type),
        n_trials=n_trials
    )
    
    best_params = study.best_params
    best_score = study.best_value
    
    print(f"Optimization completed in {(time.time() - start_time)/60:.2f} minutes")
    print(f"Best AUC: {best_score:.6f}")
    print("Best parameters:", best_params)
    
    # Train final model with best parameters
    if model_type == 'xgb':
        final_model = xgb.XGBRegressor(**best_params)
    else:
        final_model = lgb.LGBMRegressor(**best_params)
    
    print("Training final model on all training data...")
    final_model.fit(X, y)
    
    # Save model
    model_filename = f"models/{model_type}_{feature_selection}_model.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(final_model, f)
    print(f"Model saved to {model_filename}")
    
    # Generate predictions for test set
    print("Generating predictions for test set...")
    test_features = test_df.drop(['id'], axis=1, errors='ignore')
    test_predictions = final_model.predict(test_features)
    
    # Create submission file
    submission = pd.DataFrame({
        'id': test_df['id'],
        'rainfall': test_predictions
    })
    
    submission_filename = f"data/submission_{model_type}_{feature_selection}.csv"
    submission.to_csv(submission_filename, index=False)
    print(f"Predictions saved to {submission_filename}")
    
    # Save feature names for analysis
    pd.DataFrame({'feature': X.columns}).to_csv(
        f'models/feature_names_{feature_selection}.csv', index=False)
    
    return final_model, best_score, best_params

if __name__ == "__main__":
    # Run optimization for XGBoost with Boruta top features
    model, score, params = optimize_and_train(
        feature_selection='boruta_top',
        model_type='xgb',
        n_trials=50
    )
    
    print(f"\nFinal model score: {score:.6f}")
