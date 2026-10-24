import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import pickle
from datetime import datetime

def generate_submission(strategy, model_params=None, n_folds=5):
    """
    Generate submission for a specific strategy and save with score in filename
    
    Args:
        strategy: Feature engineering strategy ('all', 'minimal', 'boruta_top')
        model_params: Dictionary of model parameters (optional)
        n_folds: Number of cross-validation folds
    
    Returns:
        score: The cross-validation score
    """
    from data_loader import load_data, preprocess_data
    from feature_engineering import engineer_features
    
    print(f"\n{'='*50}")
    print(f"Generating submission for strategy: {strategy}")
    print(f"{'='*50}")
    
    # Load and preprocess data
    train_df, test_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # Apply feature engineering with specified strategy
    train_df, test_df = engineer_features(train_df, test_df, strategy=strategy)
    
    # Remove categorical columns that have been one-hot encoded
    if 'season' in train_df.columns:
        train_df = train_df.drop('season', axis=1)
    if 'season' in test_df.columns:
        test_df = test_df.drop('season', axis=1)
    
    # Prepare features and target
    X = train_df.drop(['id', 'rainfall'], axis=1, errors='ignore')
    y = train_df['rainfall']
    
    # Default model parameters
    if model_params is None:
        model_params = {
            'n_estimators': 500,
            'learning_rate': 0.03,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
    
    # Initialize cross-validation
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize arrays for oof and test predictions
    oof_preds = np.zeros(X.shape[0])
    test_preds = np.zeros(test_df.shape[0])
    
    fold_scores = []
    
    # Create directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f'models/{strategy}_{timestamp}'
    os.makedirs(run_dir, exist_ok=True)
    
    # Train models using cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"Training fold {fold+1}/{n_folds}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train XGBoost model
        model = xgb.XGBRegressor(**model_params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            early_stopping_rounds=50,
            verbose=100
        )
        
        # Save fold model
        pickle.dump(model, open(f'{run_dir}/xgb_fold_{fold}.pkl', 'wb'))
        
        # Make predictions for this fold
        fold_preds = model.predict(X_val)
        oof_preds[val_idx] = fold_preds
        
        # Calculate fold score
        fold_auc = roc_auc_score(y_val, fold_preds)
        fold_scores.append(fold_auc)
        print(f"Fold {fold+1} AUC: {fold_auc:.6f}")
        
        test_preds += model.predict(
            test_df.drop(['id'], axis=1, errors='ignore')
        ) / n_folds
    
    # Calculate overall CV score
    cv_auc = roc_auc_score(y, oof_preds)
    print(f"\nCross-Validation Results:")
    print(f"Individual fold AUC scores: {[round(score, 6) for score in fold_scores]}")
    print(f"Mean fold AUC: {np.mean(fold_scores):.6f}")
    print(f"Overall CV AUC: {cv_auc:.6f}")
    
    # Format the score as a string for filename
    score_str = f"{cv_auc:.6f}".replace('.', '_')
    
    # Create submission with mean predictions from all folds
    submission = pd.DataFrame({
        'id': test_df['id'],
        'rainfall': test_preds
    })
    
    # Save submission file with strategy and score in filename
    submission_path = f'data/submission_{strategy}_score_{score_str}.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to '{submission_path}'")
    
    # Also save a copy to the current submission.csv file
    submission.to_csv('data/submission.csv', index=False)
    
    # Return the score
    return cv_auc

if __name__ == "__main__":
    # List of strategies to try
    strategies = ['all', 'minimal', 'boruta_top']
    
    # Custom model parameters to try
    model_configs = [
        {
            'name': 'default',
            'params': None  # Use defaults
        },
        {
            'name': 'deeper',
            'params': {
                'n_estimators': 500,
                'learning_rate': 0.03,
                'max_depth': 8,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        },
        {
            'name': 'more_trees',
            'params': {
                'n_estimators': 800,
                'learning_rate': 0.02,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        }
    ]
    
    results = []
    
    # Generate submissions for each strategy with default parameters
    for strategy in strategies:
        score = generate_submission(strategy)
        results.append({
            'strategy': strategy,
            'model_config': 'default',
            'score': score
        })
    
    # Try additional model configurations for best strategy
    best_strategy = max(results, key=lambda x: x['score'])['strategy']
    print(f"\nBest strategy found: {best_strategy}")
    
    for config in model_configs[1:]:  # Skip default which was already run
        print(f"Testing {best_strategy} with {config['name']} configuration")
        score = generate_submission(f"{best_strategy}_{config['name']}", config['params'])
        results.append({
            'strategy': best_strategy,
            'model_config': config['name'],
            'score': score
        })
    
    # Print summary of all results
    print("\n" + "="*50)
    print("SUMMARY OF ALL SUBMISSIONS")
    print("="*50)
    
    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    
    for i, result in enumerate(results):
        print(f"{i+1}. Strategy: {result['strategy']}, "
              f"Config: {result['model_config']}, "
              f"Score: {result['score']:.6f}")
