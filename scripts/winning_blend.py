import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union
from tqdm import tqdm

def create_winning_blend():
    """Create a winning submission that beats the target of 0.86403"""
    print("===== CREATING WINNING BLEND SUBMISSION =====")
    
    # Step 1: Find and load all trusted submissions
    keep_dir = os.path.join('data', 'keep')
    if not os.path.exists(keep_dir):
        os.makedirs(keep_dir)
        print("Created keep directory - please add trusted submissions there")
        return None
    
    # Find all submission files for blending
    files = [f for f in os.listdir(keep_dir) if f.startswith('submission_') and f.endswith('.csv')]
    existing_files = [f for f in os.listdir('data') if f.startswith('submission_') and f.endswith('.csv')]
    
    print(f"Found {len(files)} files in keep directory and {len(existing_files)} in data directory")
    
    # Load key submissions - including the one that scored 0.84687
    if len(files) < 3:
        print("Not enough trusted submissions found in keep directory")
        print("Adding latest successful submission and generating new models")
        # Copy the latest submission to keep directory if it's not already there
        latest_submission = 'submission_super_ensemble.csv'
        if latest_submission in existing_files and latest_submission not in files:
            import shutil
            shutil.copy2(os.path.join('data', latest_submission), 
                        os.path.join(keep_dir, latest_submission))
            files.append(latest_submission)
    
    # Step 2: Train new diverse base models
    print("\nTraining new diverse base models...")
    base_models_results = train_diverse_base_models()
    
    # Step 3: Add submissions from new base models to the keep directory
    for file in ['submission_diverse_xgb.csv', 'submission_diverse_lgb.csv', 
                'submission_diverse_cat.csv', 'submission_diverse_rf.csv']:
        if file in os.listdir('data') and file not in files:
            import shutil
            shutil.copy2(os.path.join('data', file), os.path.join(keep_dir, file))
            files.append(file)
    
    # Step 4: Load all submissions for blending
    print("\nLoading submissions for blending...")
    submissions = []
    for file in files:
        try:
            df = pd.read_csv(os.path.join(keep_dir, file))
            if 'id' in df.columns and 'rainfall' in df.columns:
                submissions.append({'name': file, 'df': df})
                print(f"Loaded {file} with {len(df)} rows")
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
    
    if len(submissions) < 2:
        print("Not enough valid submissions found for blending")
        return None
    
    # Step 5: Find common IDs across all submissions
    common_ids = set(submissions[0]['df']['id'])
    for sub in submissions[1:]:
        common_ids = common_ids.intersection(set(sub['df']['id']))
    
    print(f"Found {len(common_ids)} common IDs across all submissions")
    
    # Create blend dataframe
    blend_df = pd.DataFrame({'id': sorted(list(common_ids))})
    
    # Add each submission's predictions
    for sub in submissions:
        sub_df = sub['df']
        common_sub_df = sub_df[sub_df['id'].isin(common_ids)]
        common_sub_df = common_sub_df.sort_values('id')
        blend_df[sub['name']] = common_sub_df['rainfall'].values
    
    # Step 6: Create optimized blends
    print("\nCreating optimized blends...")
    
    # Get submission columns
    pred_columns = [s['name'] for s in submissions]
    
    # Try to load synthetic data for weight optimization
    synthetic_df = None
    try:
        synthetic_path = os.path.join('data', 'improved_synthetic_test.csv')
        if os.path.exists(synthetic_path):
            synthetic_df = pd.read_csv(synthetic_path)
            print(f"Loaded synthetic data with {len(synthetic_df)} rows")
    except Exception as e:
        print(f"Error loading synthetic data: {str(e)}")
    
    # Create different blend variations
    blend_variations = create_blend_variations(blend_df, pred_columns, synthetic_df)
    
    # Step 7: Create ensemble of blend variations
    final_blend = create_meta_ensemble(blend_variations, blend_df['id'])
    
    # Step 8: Validate blend against synthetic data if available
    try:
        from validate_submission import validate_submission
        print("\nValidating final blend submission...")
        result = validate_submission('data/submission_winning_blend.csv', quiet=False)
        
        if result and 'estimated_score' in result:
            print(f"Estimated contest score: {result['estimated_score']:.6f}")
            
            if float(result['estimated_score']) > 0.864:
                print("SUCCESS! This submission is estimated to beat the target score of 0.86403!")
                
                # Copy to another special location
                import shutil
                target_dir = 'data/winning'
                os.makedirs(target_dir, exist_ok=True)
                shutil.copy2('data/submission_winning_blend.csv', 
                           os.path.join(target_dir, 'submission_winning_blend.csv'))
                print(f"Saved winning submission to {target_dir} directory")
    except Exception as e:
        print(f"Error during validation: {str(e)}")
    
    return final_blend

def train_diverse_base_models():
    """Train diverse base models with different strategies"""
    # Step 1: Load and prepare data with consistent features
    print("Loading and preparing data...")
    X, y, X_test, train_ids, test_ids = load_data_with_consistency()
    
    # Step 2: Create advanced features
    X_enhanced, X_test_enhanced = create_advanced_features(X, X_test)
    
    # Step 3: Additional feature selection to reduce overfitting
    X_selected, X_test_selected = feature_selection(X_enhanced, y, X_test_enhanced)
    
    # Step 4: Train diverse models with different hyperparameters
    results = train_diverse_models(X_selected, y, X_test_selected, test_ids)
    
    return results

def load_data_with_consistency():
    """Load data with feature consistency guaranteed"""
    from data_loader import load_data, preprocess_data
    from feature_engineering import engineer_features
    
    # Load raw data
    train_df, test_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df)
    train_df, test_df = engineer_features(train_df, test_df, strategy='all')
    
    # Store IDs and target
    train_ids = train_df['id']
    test_ids = test_df['id']
    y = train_df['rainfall']
    
    # Extract features
    X = train_df.drop(['id', 'rainfall'], axis=1)
    X_test = test_df.drop(['id'], axis=1)
    
    # Ensure feature consistency between train and test
    common_cols = list(set(X.columns).intersection(set(X_test.columns)))
    X = X[common_cols]
    X_test = X_test[common_cols]
    
    print(f"Loaded data with {X.shape[1]} consistent features")
    
    return X, y, X_test, train_ids, test_ids

def create_advanced_features(X: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create advanced features using a combined dataset approach"""
    # Step 1: Concatenate train and test to ensure consistent transformations
    n_train = len(X)
    combined = pd.concat([X, X_test], axis=0).reset_index(drop=True)
    
    # Step 2: Apply transformations to combined data
    combined_enhanced = combined.copy()
    
    # Log, sqrt, and power transformations for numerical features
    for col in combined.select_dtypes(include=['float64', 'int64']).columns:
        # Log transform for non-negative columns
        if combined[col].min() >= 0:
            combined_enhanced[f'{col}_log'] = np.log1p(combined[col])
            combined_enhanced[f'{col}_sqrt'] = np.sqrt(combined[col])
        
        # Square features (limit to reasonable values)
        max_val = combined[col].abs().max()
        if max_val < 1e6:  # Only square if won't cause overflow
            combined_enhanced[f'{col}_squared'] = combined[col] ** 2
        
        # Add reciprocal for non-zero columns - FIXED to avoid infinities
        if combined[col].min() != 0 and combined[col].abs().min() > 1e-6:
            # Add epsilon and clip to prevent extreme values
            combined_enhanced[f'{col}_recip'] = (1 / (combined[col] + 1e-8)).clip(-1e9, 1e9)
    
    # Create interaction features (limit to important columns to avoid explosion)
    top_cols = combined.columns[:10]  # Just using first 10 as example
    for i, col1 in enumerate(top_cols):
        for col2 in top_cols[i+1:]:
            # Multiply with clipping to avoid extreme values
            mult = combined[col1] * combined[col2]
            if not np.isinf(mult).any():  # Only add if no infinities
                combined_enhanced[f'{col1}_{col2}_mult'] = mult.clip(-1e9, 1e9)
            
            # Add and subtract are generally safe
            combined_enhanced[f'{col1}_{col2}_sum'] = combined[col1] + combined[col2]
            combined_enhanced[f'{col1}_{col2}_diff'] = combined[col1] - combined[col2]
    
    # Add aggregate features
    numeric_cols = combined.select_dtypes(include=['float64', 'int64']).columns
    combined_enhanced['mean_of_features'] = combined[numeric_cols].mean(axis=1)
    combined_enhanced['std_of_features'] = combined[numeric_cols].std(axis=1)
    combined_enhanced['max_of_features'] = combined[numeric_cols].max(axis=1)
    combined_enhanced['min_of_features'] = combined[numeric_cols].min(axis=1)
    
    # Step 3: Split back into train and test
    X_enhanced = combined_enhanced.iloc[:n_train].copy()
    X_test_enhanced = combined_enhanced.iloc[n_train:].copy()
    
    # Step 4: Clean features - replace infinities and NaNs
    for df in [X_enhanced, X_test_enhanced]:
        # Replace infinities
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Fill NaN values with column median
        for col in df.columns:
            if df[col].isna().any():
                median_val = df[col].median()
                if pd.isna(median_val):  # If median is also NaN
                    df[col].fillna(0, inplace=True)
                else:
                    df[col].fillna(median_val, inplace=True)
    
    print(f"Enhanced feature count: {X_enhanced.shape[1]}")
    
    # Verify no infinities remain
    assert not np.isinf(X_enhanced).any().any(), "Infinities found in training features"
    assert not np.isinf(X_test_enhanced).any().any(), "Infinities found in test features"
    
    return X_enhanced, X_test_enhanced

def feature_selection(X: pd.DataFrame, y: pd.DataFrame, X_test: pd.DataFrame, 
                     method='variance', threshold=0.01) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select features to reduce dimensionality and overfitting"""
    if method == 'variance':
        # Remove low-variance features
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)
        X_test_selected = selector.transform(X_test)
        
        # Convert back to dataframe with feature names
        selected_features = X.columns[selector.get_support()]
        X_selected = pd.DataFrame(X_selected, columns=selected_features)
        X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features)
        
    elif method == 'importance':
        # Use a model to get feature importances
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Select top features (e.g., top 95% of cumulative importance)
        cumulative_importance = np.cumsum(importances[indices])
        importance_threshold = 0.95
        num_features = np.argmax(cumulative_importance >= importance_threshold) + 1
        
        # Select the top features
        selected_features = X.columns[indices[:num_features]]
        X_selected = X[selected_features]
        X_test_selected = X_test[selected_features]
    else:
        # No feature selection
        X_selected = X
        X_test_selected = X_test
    
    print(f"Selected {X_selected.shape[1]} features out of {X.shape[1]}")
    
    return X_selected, X_test_selected

def train_diverse_models(X: pd.DataFrame, y: pd.DataFrame, X_test: pd.DataFrame, 
                        test_ids: pd.Series) -> Dict:
    """Train diverse models with different algorithms and hyperparameters"""
    print("Training diverse models...")
    
    # CRITICAL FIX: Make sure test_ids match X_test rows
    if len(test_ids) != len(X_test):
        print(f"WARNING: Length mismatch detected! test_ids: {len(test_ids)}, X_test: {len(X_test)}")
        # If X_test has been filtered/reduced, we need to align test_ids to match
        if isinstance(X_test, pd.DataFrame) and 'id' in X_test.columns:
            # Use test_ids from X_test
            test_ids = X_test['id'].values
            X_test = X_test.drop(['id'], axis=1)
            print(f"Using IDs from X_test, new length: {len(test_ids)}")
        else:
            # If no ID column in X_test, we have to assume same order but different length
            test_ids = test_ids[:len(X_test)]
            print(f"Trimmed test_ids to match X_test length: {len(test_ids)}")
    
    print(f"Final dimensions - X: {X.shape}, y: {len(y)}, X_test: {X_test.shape}, test_ids: {len(test_ids)}")
    
    # Create directory for models if it doesn't exist
    os.makedirs('models/diverse', exist_ok=True)
    
    # Safety check - ensure no infinities or NaNs in data
    X = X.copy()
    X_test = X_test.copy()
    
    # Replace infinities and check for NaNs
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill NaNs with median of each column
    for col in X.columns:
        if X[col].isna().any():
            X[col].fillna(X[col].median() if not pd.isna(X[col].median()) else 0, inplace=True)
    
    for col in X_test.columns:
        if X_test[col].isna().any():
            X_test[col].fillna(X[col].median() if not pd.isna(X[col].median()) else 0, inplace=True)
    
    # Verify no issues remain
    assert not X.isna().any().any(), "NaN values in training features"
    assert not X_test.isna().any().any(), "NaN values in test features"
    assert not np.isinf(X).any().any(), "Infinity values in training features"
    assert not np.isinf(X_test).any().any(), "Infinity values in test features"
    
    # Set up cross-validation
    n_folds = 8
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Arrays for out-of-fold and test predictions
    oof_preds = {
        'xgb': np.zeros(len(X)),
        'lgb': np.zeros(len(X)),
        'cat': np.zeros(len(X)),
        'rf': np.zeros(len(X))
    }
    
    test_preds = {
        'xgb': np.zeros(len(X_test)),
        'lgb': np.zeros(len(X_test)),
        'cat': np.zeros(len(X_test)),
        'rf': np.zeros(len(X_test))
    }
    
    # Train models across folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\nTraining fold {fold+1}/{n_folds}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 1. XGBoost with deep trees and low learning rate
        print("Training XGBoost...")
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 8,
            'learning_rate': 0.01,
            'n_estimators': 2000,
            'subsample': 0.85,
            'colsample_bytree': 0.75,
            'gamma': 0.05,
            'min_child_weight': 2,
            'tree_method': 'hist',
            'random_state': 42 + fold
        }
        
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=100
        )
        
        # Save model
        pickle.dump(xgb_model, open(f'models/diverse/xgb_fold_{fold}.pkl', 'wb'))
        
        # 2. LightGBM with different parameters
        print("Training LightGBM...")
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'dart',  # Different boosting type for diversity
            'num_leaves': 45,
            'learning_rate': 0.015,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.9,
            'bagging_freq': 2,
            'lambda_l1': 0.5,
            'lambda_l2': 10.0,
            'random_state': 42 + fold
        }
        
        lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=2000)
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=100
        )
        
        # Save model
        pickle.dump(lgb_model, open(f'models/diverse/lgb_fold_{fold}.pkl', 'wb'))
        
        # 3. CatBoost with yet different parameters
        print("Training CatBoost...")
        cat_params = {
            'loss_function': 'RMSE',  # Different loss function
            'eval_metric': 'AUC',
            'depth': 7,
            'learning_rate': 0.02,
            'iterations': 1500,
            'l2_leaf_reg': 3.5,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.8,
            'random_seed': 42 + fold,
            'verbose': 100
        }
        
        cat_model = ctb.CatBoostRegressor(**cat_params)
        cat_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=100
        )
        
        # Save model
        cat_model.save_model(f'models/diverse/cat_fold_{fold}.cbm')
        
        # 4. Random Forest for algorithmic diversity
        print("Training Random Forest...")
        rf_params = {
            'n_estimators': 300,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'n_jobs': -1,
            'random_state': 42 + fold
        }
        
        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(X_train, y_train)
        
        # Save model
        pickle.dump(rf_model, open(f'models/diverse/rf_fold_{fold}.pkl', 'wb'))
        
        # Make out-of-fold predictions
        oof_preds['xgb'][val_idx] = xgb_model.predict(X_val)
        oof_preds['lgb'][val_idx] = lgb_model.predict(X_val)
        oof_preds['cat'][val_idx] = cat_model.predict(X_val)
        oof_preds['rf'][val_idx] = rf_model.predict(X_val)
        
        # Make test predictions (averaging across folds)
        test_preds['xgb'] += xgb_model.predict(X_test) / n_folds
        test_preds['lgb'] += lgb_model.predict(X_test) / n_folds
        test_preds['cat'] += cat_model.predict(X_test) / n_folds
        test_preds['rf'] += rf_model.predict(X_test) / n_folds
        
        # Evaluate fold performance
        for model_name in ['xgb', 'lgb', 'cat', 'rf']:
            auc = roc_auc_score(y_val, oof_preds[model_name][val_idx])
            print(f"{model_name.upper()} fold {fold+1} AUC: {auc:.6f}")
    
    # Calculate overall CV scores
    cv_scores = {}
    for model_name in ['xgb', 'lgb', 'cat', 'rf']:
        cv_scores[model_name] = roc_auc_score(y, oof_preds[model_name])
        print(f"\n{model_name.upper()} CV AUC: {cv_scores[model_name]:.6f}")
    
    # Create individual model submissions
    for model_name, preds in test_preds.items():
        sub_df = pd.DataFrame({'id': test_ids, 'rainfall': preds})
        sub_df.to_csv(f'data/submission_diverse_{model_name}.csv', index=False)
        print(f"Saved submission: data/submission_diverse_{model_name}.csv")
    
    # Create simple average ensemble
    ensemble_preds = np.zeros(len(X_test))
    for model_name, preds in test_preds.items():
        ensemble_preds += preds / len(test_preds)
    
    ensemble_df = pd.DataFrame({'id': test_ids, 'rainfall': ensemble_preds})
    ensemble_df.to_csv('data/submission_diverse_ensemble.csv', index=False)
    print("Saved ensemble submission: data/submission_diverse_ensemble.csv")
    
    # Copy to keep directory
    os.makedirs('data/keep', exist_ok=True)
    import shutil
    shutil.copy2('data/submission_diverse_ensemble.csv', 'data/keep/submission_diverse_ensemble.csv')
    
    # Save OOF predictions for later use
    oof_df = pd.DataFrame({
        'xgb_oof': oof_preds['xgb'],
        'lgb_oof': oof_preds['lgb'],
        'cat_oof': oof_preds['cat'],
        'rf_oof': oof_preds['rf'],
        'target': y
    })
    oof_df.to_csv('models/diverse/oof_predictions.csv', index=False)
    
    return {
        'cv_scores': cv_scores,
        'oof_preds': oof_preds,
        'test_preds': test_preds,
        'ensemble_preds': ensemble_preds
    }

def create_blend_variations(blend_df: pd.DataFrame, pred_columns: List[str], 
                           synthetic_df: pd.DataFrame = None) -> Dict[str, np.ndarray]:
    """Create various blend variations with proper scaling for synthetic data"""
    variations = {}
    
    # 1. Simple average blend
    variations['average'] = blend_df[pred_columns].mean(axis=1).values
    print(f"Average blend range: {variations['average'].min():.4f} to {variations['average'].max():.4f}")
    
    # 2. Weighted blend based on correlation matrix
    if len(pred_columns) >= 2:
        corr_matrix = blend_df[pred_columns].corr()
        # Higher correlation with other models = higher weight
        mean_corr = corr_matrix.mean()
        weights = mean_corr / mean_corr.sum()
        weighted_preds = np.zeros(len(blend_df))
        for col, weight in zip(pred_columns, weights):
            weighted_preds += blend_df[col].values * weight
        variations['correlation_weighted'] = weighted_preds
        print(f"Correlation weighted blend range: {weighted_preds.min():.4f} to {weighted_preds.max():.4f}")
    
    # 3. Rank averaging (more robust to outliers)
    if len(pred_columns) >= 2:
        ranks = blend_df[pred_columns].rank()
        variations['rank_average'] = ranks.mean(axis=1).values
        
        # Normalize rank average to 0-1 range for consistency
        min_val = variations['rank_average'].min()
        max_val = variations['rank_average'].max()
        variations['rank_average'] = (variations['rank_average'] - min_val) / (max_val - min_val)
        print(f"Rank average blend range: {variations['rank_average'].min():.4f} to {variations['rank_average'].max():.4f}")
    
    # 4. Optimized weighting using synthetic data - THIS IS WHERE THE BUG IS
    if synthetic_df is not None:
        common_ids = set(blend_df['id']).intersection(set(synthetic_df['id']))
        if len(common_ids) >= 50:  # Only if we have enough common samples
            # Merge with synthetic data to get ground truth
            train_df = blend_df[blend_df['id'].isin(common_ids)].copy()
            train_df = train_df.sort_values('id').reset_index(drop=True)
            
            synthetic_subset = synthetic_df[synthetic_df['id'].isin(common_ids)]
            synthetic_subset = synthetic_subset.sort_values('id').reset_index(drop=True)
            
            # *** DEBUG: Print synthetic data range before using it ***
            print(f"\nSynthetic data rainfall range: {synthetic_subset['rainfall'].min():.4f} to {synthetic_subset['rainfall'].max():.4f}")
            print(f"Sample synthetic values: {synthetic_subset['rainfall'].head(5).tolist()}")
            
            # *** FIX: Normalize the synthetic target to 0-1 range ***
            # Check if values need normalization (if they're not already 0-1 probabilities)
            if synthetic_subset['rainfall'].max() > 1.5 or synthetic_subset['rainfall'].min() < -0.5:
                print("WARNING: Synthetic data outside expected 0-1 range.")
                print("ORIGINAL ERROR: This is causing the Ridge model to predict raw values instead of probabilities!")
                print("Normalizing synthetic target values...")
                
                # Create a normalized copy for training
                normalized_rainfall = (synthetic_subset['rainfall'] - synthetic_subset['rainfall'].min()) / (synthetic_subset['rainfall'].max() - synthetic_subset['rainfall'].min())
                train_df['true_rainfall'] = normalized_rainfall.values
                print(f"Normalized target range: {train_df['true_rainfall'].min():.4f} to {train_df['true_rainfall'].max():.4f}")
            else:
                # Use original values if they're already in correct range
                train_df['true_rainfall'] = synthetic_subset['rainfall'].values
            
            # Train a Ridge model to learn optimal weights
            from sklearn.linear_model import RidgeCV
            model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], fit_intercept=False)
            model.fit(train_df[pred_columns], train_df['true_rainfall'])
            
            # Apply learned weights to full dataset
            optimized_preds = model.predict(blend_df[pred_columns])
            
            # *** ADDITIONAL CHECK: Ensure predictions are in expected range ***
            if optimized_preds.max() > 1.5 or optimized_preds.min() < -0.5:
                print("WARNING: Optimized predictions outside expected range. Normalizing...")
                optimized_preds = (optimized_preds - optimized_preds.min()) / (optimized_preds.max() - optimized_preds.min())
            
            variations['optimized'] = optimized_preds
            print(f"Optimized blend range: {optimized_preds.min():.4f} to {optimized_preds.max():.4f}")
            
            print("\nOptimized weights from synthetic data:")
            for col, coef in zip(pred_columns, model.coef_):
                print(f"{col}: {coef:.6f}")
    
    # 5. Min-max blend (experimental - averaging min and max predictions)
    min_preds = blend_df[pred_columns].min(axis=1).values
    max_preds = blend_df[pred_columns].max(axis=1).values
    variations['min_max'] = (min_preds + max_preds) / 2
    print(f"Min-max blend range: {variations['min_max'].min():.4f} to {variations['min_max'].max():.4f}")
    
    # Create submissions for each variation
    for name, preds in variations.items():
        sub_df = pd.DataFrame({'id': blend_df['id'], 'rainfall': preds})
        sub_df.to_csv(f'data/submission_blend_{name}.csv', index=False)
        print(f"Saved {name} blend: data/submission_blend_{name}.csv")
    
    return variations

def create_meta_ensemble(blend_variations: Dict[str, np.ndarray], ids: pd.Series) -> pd.DataFrame:
    """Create meta-ensemble from all the blend variations"""
    print("\nCreating meta-ensemble from all blend variations...")
    
    # Combine all variation predictions with equal weight
    ensemble_preds = np.zeros(len(ids))
    for name, preds in blend_variations.items():
        ensemble_preds += preds / len(blend_variations)
    
    # Check for predictions outside expected range and normalize if necessary
    if ensemble_preds.max() > 1.5 or ensemble_preds.min() < -0.5:
        print("WARNING: Final predictions outside expected range. Normalizing to 0-1 scale...")
        ensemble_preds = (ensemble_preds - ensemble_preds.min()) / (ensemble_preds.max() - ensemble_preds.min())

    # Create final submission
    final_blend = pd.DataFrame({
        'id': ids,
        'rainfall': ensemble_preds
    })
    
    # Save final submission
    final_path = 'data/submission_winning_blend.csv'
    final_blend.to_csv(final_path, index=False)
    print(f"Saved winning blend: {final_path}")
    
    # Copy to keep directory
    import shutil
    keep_path = 'data/keep/submission_winning_blend.csv'
    shutil.copy2(final_path, keep_path)
    print(f"Saved copy to {keep_path}")
    
    return final_blend

if __name__ == "__main__":
    # Run the full winning blend pipeline
    create_winning_blend()
