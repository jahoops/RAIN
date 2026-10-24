import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def load_data_with_consistency():
    """Load and prepare data with guaranteed feature consistency"""
    from data_loader import load_data, preprocess_data
    from feature_engineering import engineer_features
    
    print("Loading and preprocessing data...")
    train_df, test_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df)
    train_df, test_df = engineer_features(train_df, test_df, strategy='all')
    
    # Store IDs and target
    train_ids = train_df['id']
    test_ids = test_df['id']
    y = train_df['rainfall']
    
    # Get features only (drop non-feature columns)
    X = train_df.drop(['id', 'rainfall'], axis=1)
    X_test = test_df.drop(['id'], axis=1)
    
    # Ensure feature consistency - use only common columns
    common_cols = list(set(X.columns).intersection(set(X_test.columns)))
    X = X[common_cols].copy()
    X_test = X_test[common_cols].copy()
    
    # Verify columns match
    assert X.shape[1] == X_test.shape[1], "Column count mismatch!"
    
    print(f"Loaded data with {X.shape[1]} consistent features")
    print(f"Train shape: {X.shape}")
    print(f"Test shape: {X_test.shape}")
    
    return X, y, X_test, train_ids, test_ids

def create_advanced_features(X_train, X_test):
    """Create advanced features with guaranteed consistency between train and test"""
    print("Creating advanced features...")
    
    # Start with a guaranteed consistent dataset
    n_train = len(X_train)
    combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    
    # Feature group 1: Logarithmic and power transformations
    for col in tqdm(combined.select_dtypes(include=['float64', 'int64']).columns):
        if combined[col].min() >= 0:
            # Log transform (with 1 offset to avoid log(0))
            combined[f'{col}_log'] = np.log1p(combined[col])
            # Square root 
            combined[f'{col}_sqrt'] = np.sqrt(combined[col])
        
        # Square and cube - capture non-linear relationships
        combined[f'{col}_squared'] = combined[col] ** 2
        combined[f'{col}_cubed'] = combined[col] ** 3
        
        # Reciprocal transformation (with safety for zeros)
        if combined[col].min() > 0:
            combined[f'{col}_recip'] = 1 / combined[col]
    
    # Feature group 2: Interactions between high-importance features
    # Use first 20 columns as proxy for important features
    important_cols = combined.columns[:20]
    for i, col1 in enumerate(important_cols):
        for col2 in important_cols[i+1:i+6]:  # Limit to avoid explosion of features
            combined[f'{col1}_{col2}_sum'] = combined[col1] + combined[col2]
            combined[f'{col1}_{col2}_diff'] = combined[col1] - combined[col2]
            combined[f'{col1}_{col2}_mult'] = combined[col1] * combined[col2]
            # Ratio with safety for division by zero
            combined[f'{col1}_{col2}_div'] = combined[col1] / (combined[col2] + 1e-5)
    
    # Feature group 3: Statistical aggregations
    numeric_cols = combined.select_dtypes(include=['float64', 'int64']).columns[:30]
    combined['mean_of_top15'] = combined[numeric_cols[:15]].mean(axis=1)
    combined['std_of_top15'] = combined[numeric_cols[:15]].std(axis=1)
    combined['max_of_top15'] = combined[numeric_cols[:15]].max(axis=1)
    combined['min_of_top15'] = combined[numeric_cols[:15]].min(axis=1)
    
    # Feature group 4: Row statistics
    combined['row_mean'] = combined.mean(axis=1)
    combined['row_std'] = combined.std(axis=1)
    combined['row_min'] = combined.min(axis=1)
    combined['row_max'] = combined.max(axis=1)
    combined['row_range'] = combined.max(axis=1) - combined.min(axis=1)
    
    # Split back into train and test datasets
    X_train_enhanced = combined.iloc[:n_train].copy()
    X_test_enhanced = combined.iloc[n_train:].copy()
    
    print(f"Feature count: {X_train.shape[1]} → {X_train_enhanced.shape[1]}")
    
    # Final consistency check
    assert X_train_enhanced.shape[1] == X_test_enhanced.shape[1], "Column count mismatch!"
    assert all(X_train_enhanced.columns == X_test_enhanced.columns), "Column order mismatch!"
    
    return X_train_enhanced, X_test_enhanced

def train_multi_model_ensemble(X, y, X_test, test_ids, n_folds=10):
    """Train a powerful ensemble of multiple model types with extensive CV"""
    print(f"\nTraining multi-model ensemble with {n_folds}-fold CV")
    
    # Output directories
    os.makedirs('models/winning', exist_ok=True)
    
    # Initialize cross-validation
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize arrays for out-of-fold and test predictions
    oof_preds = {
        'xgb': np.zeros(len(X)),
        'lgb': np.zeros(len(X)),
        'cat': np.zeros(len(X))
    }
    
    test_preds = {
        'xgb': np.zeros(len(X_test)),
        'lgb': np.zeros(len(X_test)),
        'cat': np.zeros(len(X_test))
    }
    
    # Track model performances
    cv_scores = {
        'xgb': [],
        'lgb': [],
        'cat': []
    }
    
    # Train models using cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n--- Training fold {fold+1}/{n_folds} ---")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 1. Train XGBoost model
        print("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            objective='binary:logistic',
            eval_metric='auc',
            max_depth=5,
            learning_rate=0.01,
            n_estimators=3000,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=3,
            gamma=0.01,
            random_state=42+fold
        )
        
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=100
        )
        
        pickle.dump(xgb_model, open(f'models/winning/xgb_fold_{fold}.pkl', 'wb'))
        
        # 2. Train LightGBM model
        print("\nTraining LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            objective='binary',
            metric='auc',
            learning_rate=0.01,
            n_estimators=3000,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42+fold
        )
        
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=100
        )
        
        pickle.dump(lgb_model, open(f'models/winning/lgb_fold_{fold}.pkl', 'wb'))
        
        # 3. Train CatBoost model (with more conservative parameters for stability)
        print("\nTraining CatBoost...")
        cat_model = ctb.CatBoostRegressor(
            objective='RMSE',
            eval_metric='AUC',
            iterations=2000,
            learning_rate=0.01,
            depth=6,
            l2_leaf_reg=3,
            random_seed=42+fold,
            bootstrap_type='Bernoulli',
            subsample=0.8,
            verbose=0
        )
        
        cat_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=100
        )
        
        cat_model.save_model(f'models/winning/cat_fold_{fold}.cbm')
        
        # Make out-of-fold predictions
        oof_preds['xgb'][val_idx] = xgb_model.predict(X_val)
        oof_preds['lgb'][val_idx] = lgb_model.predict(X_val)
        oof_preds['cat'][val_idx] = cat_model.predict(X_val)
        
        # Make test predictions
        test_preds['xgb'] += xgb_model.predict(X_test) / n_folds
        test_preds['lgb'] += lgb_model.predict(X_test) / n_folds
        test_preds['cat'] += cat_model.predict(X_test) / n_folds
        
        # Calculate fold AUC for each model
        xgb_score = roc_auc_score(y_val, oof_preds['xgb'][val_idx])
        lgb_score = roc_auc_score(y_val, oof_preds['lgb'][val_idx])
        cat_score = roc_auc_score(y_val, oof_preds['cat'][val_idx])
        
        # Store scores
        cv_scores['xgb'].append(xgb_score)
        cv_scores['lgb'].append(lgb_score)
        cv_scores['cat'].append(cat_score)
        
        print(f"\nFold {fold+1} scores:")
        print(f"XGBoost AUC: {xgb_score:.6f}")
        print(f"LightGBM AUC: {lgb_score:.6f}")
        print(f"CatBoost AUC: {cat_score:.6f}")
    
    # Calculate overall CV scores
    final_cv_scores = {}
    for model_name in ['xgb', 'lgb', 'cat']:
        final_cv_scores[model_name] = roc_auc_score(y, oof_preds[model_name])
    
    print("\nFinal cross-validation scores:")
    for model_name, score in final_cv_scores.items():
        print(f"{model_name.upper()}: {score:.6f} (std: {np.std(cv_scores[model_name]):.6f})")
    
    # Create weighted ensemble
    # Calculate weights based on CV performance
    total_score = sum(final_cv_scores.values())
    weights = {model: score/total_score for model, score in final_cv_scores.items()}
    
    print("\nEnsemble weights:")
    for model, weight in weights.items():
        print(f"{model.upper()}: {weight:.6f}")
    
    # Create ensemble predictions
    ensemble_preds = (
        test_preds['xgb'] * weights['xgb'] +
        test_preds['lgb'] * weights['lgb'] +
        test_preds['cat'] * weights['cat']
    )
    
    # Create submissions
    ensemble_sub = pd.DataFrame({'id': test_ids, 'rainfall': ensemble_preds})
    
    # Also save individual model submissions
    for model in ['xgb', 'lgb', 'cat']:
        model_sub = pd.DataFrame({'id': test_ids, 'rainfall': test_preds[model]})
        model_sub.to_csv(f'data/submission_winning_{model}.csv', index=False)
        print(f"Saved {model.upper()} submission: data/submission_winning_{model}.csv")
    
    # Save ensemble submission
    ensemble_sub.to_csv('data/submission_winning_ensemble.csv', index=False)
    print("Saved ensemble submission: data/submission_winning_ensemble.csv")
    
    # Save out-of-fold predictions for analysis
    oof_df = pd.DataFrame({
        'xgb_pred': oof_preds['xgb'],
        'lgb_pred': oof_preds['lgb'],
        'cat_pred': oof_preds['cat'],
        'target': y
    })
    oof_df.to_csv('models/winning/oof_predictions.csv', index=False)
    
    # Run synthetic validation if available
    try:
        from validate_submission import validate_submission
        print("\nRunning synthetic validation...")
        results = validate_submission('data/submission_winning_ensemble.csv', quiet=False)
        
        # Save successful submission to keep directory
        if results and results.get('estimated_score', 0) >= 0.85:
            os.makedirs('data/keep', exist_ok=True)
            import shutil
            shutil.copy2('data/submission_winning_ensemble.csv', 'data/keep/')
            print("✓ Saved winning submission to keep directory")
    except Exception as e:
        print(f"Synthetic validation error: {e}")
        print("Please run validation manually.")
    
    return {
        'ensemble_preds': ensemble_preds,
        'model_preds': test_preds,
        'cv_scores': final_cv_scores,
        'weights': weights
    }

def optimize_final_blend():
    """Optimize the final blend using all previous good submissions"""
    print("\nOptimizing final submission blend...")
    
    # Look for submissions in the keep directory
    keep_dir = os.path.join('data', 'keep')
    if not os.path.exists(keep_dir):
        print("No keep directory found. Creating one.")
        os.makedirs(keep_dir)
        return None
    
    # Find all submission files
    submission_files = [f for f in os.listdir(keep_dir) if f.startswith('submission_') and f.endswith('.csv')]
    
    if len(submission_files) < 2:
        print("Not enough submissions to blend. Need at least 2 files in keep directory.")
        return None
    
    print(f"Found {len(submission_files)} submissions to blend")
    
    # Load all submissions
    submissions = []
    for file in submission_files:
        path = os.path.join(keep_dir, file)
        try:
            sub_df = pd.read_csv(path)
            if 'id' in sub_df.columns and 'rainfall' in sub_df.columns:
                submissions.append({
                    'name': file,
                    'df': sub_df
                })
                print(f"Loaded {file} with {len(sub_df)} predictions")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not submissions:
        print("No valid submissions found to blend.")
        return None
    
    # Find common IDs across all submissions
    common_ids = set(submissions[0]['df']['id'])
    for sub in submissions[1:]:
        common_ids = common_ids.intersection(set(sub['df']['id']))
    
    if not common_ids:
        print("No common IDs across submissions.")
        return None
    
    print(f"Found {len(common_ids)} common IDs across all submissions")
    
    # Create a dataframe with all predictions
    blend_df = pd.DataFrame({'id': list(common_ids)})
    
    # Add each submission's predictions
    for sub in submissions:
        sub_df = sub['df']
        common_sub_df = sub_df[sub_df['id'].isin(common_ids)].copy()
        common_sub_df = common_sub_df.sort_values('id')
        blend_df[sub['name']] = common_sub_df['rainfall'].values
    
    # Calculate default blend (average)
    prediction_cols = [sub['name'] for sub in submissions]
    blend_df['rainfall'] = blend_df[prediction_cols].mean(axis=1)
    
    # Create final blended submission
    final_blend = pd.DataFrame({
        'id': blend_df['id'],
        'rainfall': blend_df['rainfall']
    })
    
    # Save the final blend
    final_blend_path = 'data/submission_ultimate_blend.csv'
    final_blend.to_csv(final_blend_path, index=False)
    print(f"Saved final blend to {final_blend_path}")
    
    # Validate the blend
    try:
        from validate_submission import validate_submission
        print("\nValidating final blend...")
        validate_submission(final_blend_path, quiet=False)
    except Exception as e:
        print(f"Validation error: {e}")
    
    # Copy to keep directory
    import shutil
    keep_path = os.path.join('data/keep', 'submission_ultimate_blend.csv')
    shutil.copy2(final_blend_path, keep_path)
    print(f"Saved blend to {keep_path}")
    
    return final_blend

def create_winning_submission():
    """Master function to create a winning submission"""
    print("===== CREATING WINNING SUBMISSION =====\n")
    
    # Step 1: Load data with guaranteed feature consistency
    X, y, X_test, train_ids, test_ids = load_data_with_consistency()
    
    # Step 2: Create advanced features
    X_enhanced, X_test_enhanced = create_advanced_features(X, X_test)
    
    # Step 3: Train multi-model ensemble
    results = train_multi_model_ensemble(X_enhanced, y, X_test_enhanced, test_ids, n_folds=10)
    
    # Step 4: Optimize final blend with previous good submissions
    final_blend = optimize_final_blend()
    
    print("\n===== WINNING SUBMISSION CREATED =====")
    print("\nRecommended submissions to use (in order of priority):")
    print("1. data/submission_ultimate_blend.csv")
    print("2. data/submission_winning_ensemble.csv")
    print("3. data/submission_winning_xgb.csv (most stable single model)")

if __name__ == "__main__":
    create_winning_submission()
