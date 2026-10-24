import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import pickle
import os

def train_with_cv(n_folds=5, feature_strategy='all', use_synthetic=False):
    from data_loader import load_data, preprocess_data
    from feature_engineering import engineer_features
    
    # Load and preprocess data
    train_df, test_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # Apply feature engineering with specified strategy
    print(f"Using feature engineering strategy: {feature_strategy}")
    train_df, test_df = engineer_features(train_df, test_df, strategy=feature_strategy)
    
    # Remove categorical columns that have been one-hot encoded
    if 'season' in train_df.columns:
        train_df = train_df.drop('season', axis=1)
    if 'season' in test_df.columns:
        test_df = test_df.drop('season', axis=1)
    
    # Prepare features and target
    X = train_df.drop(['id', 'rainfall'], axis=1, errors='ignore')
    y = train_df['rainfall']
    
    # Initialize cross-validation
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize arrays for oof and test predictions
    oof_preds = np.zeros(X.shape[0])
    test_preds = np.zeros(test_df.shape[0])
    
    # Save feature names for analysis
    os.makedirs('models', exist_ok=True)
    pd.DataFrame({'feature': X.columns}).to_csv('models/feature_names.csv', index=False)
    
    fold_scores = []
    
    # Train models using cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"Training fold {fold+1}/{n_folds}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42+fold
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',  # Changed to AUC from RMSE
            early_stopping_rounds=50,
            verbose=100
        )
        
        # Save fold model
        pickle.dump(model, open(f'models/xgb_fold_{fold}.pkl', 'wb'))
        
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
    print(f"Overall CV AUC (estimated contest score): {cv_auc:.6f}")
    
    # Save OOF predictions for further analysis
    oof_df = pd.DataFrame({
        'id': train_df['id'],
        'true_value': y,
        'prediction': oof_preds
    })
    oof_df.to_csv('models/oof_predictions.csv', index=False)
    print("Out-of-fold predictions saved to 'models/oof_predictions.csv'")
    
    # Create submission with mean predictions from all folds
    submission = pd.DataFrame({
        'id': test_df['id'],
        'rainfall': test_preds
    })
    submission.to_csv('data/submission.csv', index=False)
    print("Predictions saved to 'data/submission.csv'")
    
    # If requested, also run synthetic data validation
    if use_synthetic:
        from validation_with_synthetic import validate_with_synthetic_data
        print("\nRunning validation with synthetic data...")
        synthetic_score = validate_with_synthetic_data(model_name='xgboost', feature_strategy=feature_strategy)
        print(f"Synthetic validation AUC: {synthetic_score:.6f}")
        
        # Calculate gap between CV and synthetic scores
        gap = cv_auc - synthetic_score
        print(f"Gap between CV and synthetic scores: {gap:.6f}")
        
        # Add both scores to experiment results
        experiment_results = pd.DataFrame({
            'CV Score': [cv_auc],
            'Synthetic Score': [synthetic_score],
            'Gap': [gap],
            'Feature Strategy': [feature_strategy]
        })
        
        # Save or append results
        experiment_results_path = 'models/experiment_results.csv'
        try:
            previous_results = pd.read_csv(experiment_results_path)
            updated_results = pd.concat([previous_results, experiment_results], ignore_index=True)
            updated_results.to_csv(experiment_results_path, index=False)
        except:
            experiment_results.to_csv(experiment_results_path, index=False)
    
    # Return the estimated contest score
    return cv_auc
    
if __name__ == "__main__":
    # Add synthetic validation option
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-synthetic', action='store_true', help='Run synthetic data validation')
    args = parser.parse_args()
    
    # Try different feature engineering strategies
    print("Training with 'all' features")
    all_features_score = train_with_cv(n_folds=5, feature_strategy='all', use_synthetic=args.use_synthetic)
    
    print("\nTraining with 'minimal' features")
    minimal_features_score = train_with_cv(n_folds=5, feature_strategy='minimal', use_synthetic=args.use_synthetic)
    
    # If you've run Boruta feature selection previously
    print("\nTraining with Boruta selected features")
    boruta_features_score = train_with_cv(n_folds=5, feature_strategy='boruta_top', use_synthetic=args.use_synthetic)
    
    # Compare results
    print("\nResults comparison:")
    print(f"All features: {all_features_score:.6f}")
    print(f"Minimal features: {minimal_features_score:.6f}")
    print(f"Boruta features: {boruta_features_score:.6f}")