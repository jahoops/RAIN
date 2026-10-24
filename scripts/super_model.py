import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import xgboost as xgb
import lightgbm as lgb
from scipy.interpolate import interp1d
import optuna
from typing import List, Dict, Tuple, Any, Union

def create_run_dir() -> str:
    """Create a new directory for storing experiment results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f'models/super_model_{timestamp}'
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def add_meteorological_features(df):
    """Add meteorologically relevant temporal features - STEP 2"""
    print("Adding meteorological seasonal features...")
    
    # Copy dataframe
    result = df.copy()
    
    # If we have month information directly
    if 'month' in result.columns:
        # Use existing month column
        pass
    # If we have day column but no month, try to extract month
    elif 'day' in result.columns and result['day'].max() > 31:
        # Assume this is day of year
        result['month'] = ((result['day'] - 1) // 30) + 1
        result['month'] = result['month'].clip(1, 12)
    # If we have date column
    elif 'date' in result.columns:
        # Extract month
        try:
            result['month'] = pd.to_datetime(result['date']).dt.month
        except:
            pass
    # If we have id column, try to derive temporal features
    elif 'id' in result.columns:
        # Assuming IDs are sequential and might represent days
        # This is speculative but worth trying
        result['pseudo_day'] = result['id'] % 365
        result['month'] = ((result['pseudo_day'] - 1) // 30) + 1
        result['month'] = result['month'].clip(1, 12)
    
    # If we now have month info, create seasons
    if 'month' in result.columns:
        # Meteorological seasons (Northern Hemisphere)
        result['season'] = result['month'].apply(lambda m: 
            0 if m in [12, 1, 2]    # Winter
            else 1 if m in [3, 4, 5] # Spring
            else 2 if m in [6, 7, 8] # Summer
            else 3)                  # Fall
        
        # Rainfall seasonality (more interpretable to trees)
        result['high_rain_season'] = result['month'].apply(
            lambda m: 1 if m in [4, 5, 6, 9, 10, 11] else 0)
        
        # Convert month to cyclical features to handle December->January transition
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
        
        # Create month indicators using one-hot encoding
        # (Tree models handle these well)
        for m in range(1, 13):
            result[f'is_month_{m}'] = (result['month'] == m).astype(int)
    
    # Create cloud interactions with season if both exist
    if 'cloud' in result.columns and 'season' in result.columns:
        # Cloud impact varies by season
        result['cloud_season'] = result['cloud'] * (result['season'] + 1)
    
    return result

def calibrate_rainfall_distribution(train_rainfall, pred_rainfall):
    """Light calibration that preserves continuous prediction values"""
    print("Applying light rainfall calibration...")
    
    # Get zero percentage
    zero_pct = (train_rainfall == 0).mean()
    print(f"Zero percentage in training: {zero_pct*100:.2f}%")
    
    # Sort predictions to find threshold
    threshold = np.percentile(pred_rainfall, zero_pct * 100)
    
    # Apply zero threshold but KEEP other values continuous
    calibrated = pred_rainfall.copy()
    calibrated[calibrated < threshold] = 0
    
    # Ensure predictions are in the valid range [0, 1]
    calibrated = np.clip(calibrated, 0, 1)
    
    # Calculate and print distribution statistics
    print(f"Calibrated predictions stats:")
    print(f"  - Mean: {calibrated.mean():.4f}")
    print(f"  - Min: {calibrated.min():.4f}, Max: {calibrated.max():.4f}")
    print(f"  - Zeros: {(calibrated == 0).sum()} ({(calibrated == 0).mean()*100:.2f}%)")
    print(f"  - Non-binary values: {((calibrated > 0) & (calibrated < 1)).sum()} " 
          f"({((calibrated > 0) & (calibrated < 1)).mean()*100:.2f}%)")
    print(f"  - Unique values: {len(np.unique(calibrated))}")
    
    return calibrated

def get_feature_sets(X: pd.DataFrame) -> Dict[str, List[str]]:
    """Create feature sets based on our GA findings"""
    feature_sets = {}
    
    # Core weather features - highest impact features based on GA results
    core_weather = ['cloud', 'humidity', 'pressure', 'windspeed', 'dewpoint', 'temperature', 
                    'maxtemp', 'mintemp']
    
    # Get available columns matching these patterns using fuzzy matching
    core_cols = []
    for feature in core_weather:
        matching_cols = [col for col in X.columns if feature.lower() in col.lower()]
        core_cols.extend(matching_cols)
    
    # Remove duplicates
    core_cols = list(set(core_cols))
    feature_sets['core_weather'] = core_cols
    
    # Cloud-focused features - cloud was repeatedly the #1 feature
    cloud_cols = [col for col in X.columns if 'cloud' in col.lower()]
    cloud_cols.extend([col for col in X.columns if col in 
                      ['humidity', 'pressure', 'temperature']])
    feature_sets['cloud_focused'] = list(set(cloud_cols))
    
    # Get all meteorological columns (exclude ids, rainfall-based features)
    all_meteo = [col for col in X.columns if not any(x in col.lower() 
                for x in ['id', 'rainfall', 'lag', 'roll'])]
    feature_sets['all_meteorological'] = all_meteo
    
    # Add all season/month columns to each feature set
    seasonal_cols = [col for col in X.columns if any(x in col.lower() for x in 
                    ['month', 'season', 'spring', 'summer', 'winter', 'fall'])]
    
    for key in feature_sets.keys():
        feature_sets[key] = list(set(feature_sets[key] + seasonal_cols))
    
    # All features
    feature_sets['all'] = X.columns.tolist()
    
    return feature_sets

def get_optimized_hyperparams(model_type):
    """Get advanced optimized hyperparameters for tree models - STEP 1"""
    
    if model_type == 'random_forest':
        return {
            'n_estimators': 1000,
            'max_depth': 15,
            'min_samples_leaf': 3,
            'max_features': 0.7,
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1
        }
    elif model_type == 'extra_trees':
        return {
            'n_estimators': 1000,
            'max_depth': 15,
            'min_samples_leaf': 3,
            'max_features': 0.7,
            'bootstrap': True,
            'random_state': 43,
            'n_jobs': -1
        }
    elif model_type == 'xgboost':
        return {
            'n_estimators': 800,
            'max_depth': 9,
            'learning_rate': 0.02,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 2,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'gamma': 0.5,
            'random_state': 44
        }
    elif model_type == 'lightgbm':
        return {
            'n_estimators': 800,
            'max_depth': 9,
            'learning_rate': 0.02,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_samples': 15,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'num_leaves': 31,
            'boosting_type': 'goss',
            'random_state': 45
        }
    
    # Default values if model type not recognized
    return {
        'n_estimators': 500,
        'max_depth': 8,
        'random_state': 42
    }

def train_advanced_models(X_train, y_train, X_val, y_val, feature_set):
    """Train optimized tree models with advanced hyperparameters - STEP 1"""
    print(f"\nTraining advanced models on {feature_set} feature set...")
    models = []
    val_predictions = []
    
    # 1. Random Forest
    print("Training Random Forest model...")
    rf_params = get_optimized_hyperparams('random_forest')
    rf_model = RandomForestRegressor(**rf_params)
    rf_model.fit(X_train, y_train)
    rf_val_pred = rf_model.predict(X_val)
    models.append(('random_forest', rf_model))
    val_predictions.append(rf_val_pred)
    rf_mse = ((rf_val_pred - y_val) ** 2).mean()
    print(f"RF validation MSE: {rf_mse:.6f}")
    
    # 2. Extra Trees
    print("Training Extra Trees model...")
    et_params = get_optimized_hyperparams('extra_trees')
    et_model = ExtraTreesRegressor(**et_params)
    et_model.fit(X_train, y_train)
    et_val_pred = et_model.predict(X_val)
    models.append(('extra_trees', et_model))
    val_predictions.append(et_val_pred)
    et_mse = ((et_val_pred - y_val) ** 2).mean()
    print(f"ET validation MSE: {et_mse:.6f}")
    
    # 3. XGBoost
    print("Training XGBoost model...")
    xgb_params = get_optimized_hyperparams('xgboost')
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train)
    xgb_val_pred = xgb_model.predict(X_val)
    models.append(('xgboost', xgb_model))
    val_predictions.append(xgb_val_pred)
    xgb_mse = ((xgb_val_pred - y_val) ** 2).mean()
    print(f"XGB validation MSE: {xgb_mse:.6f}")
    
    # 4. LightGBM
    print("Training LightGBM model...")
    lgb_params = get_optimized_hyperparams('lightgbm')
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train)
    lgb_val_pred = lgb_model.predict(X_val)
    models.append(('lightgbm', lgb_model))
    val_predictions.append(lgb_val_pred)
    lgb_mse = ((lgb_val_pred - y_val) ** 2).mean()
    print(f"LGB validation MSE: {lgb_mse:.6f}")
    
    return models, val_predictions

def create_advanced_ensemble(models, val_preds, y_val, X_test):
    """Create sophisticated ensemble from model predictions - STEP 4"""
    print("\nCreating advanced ensemble prediction...")
    
    test_predictions = []
    for model_name, model in models:
        test_pred = model.predict(X_test)
        test_predictions.append(test_pred)
    
    # Method 1: Rank-Based Ensembling
    print("Creating rank-based ensemble...")
    ranked_test_preds = []
    for pred in test_predictions:
        ranks = pd.Series(pred).rank() / len(pred)
        ranked_test_preds.append(ranks.values)
    
    rank_ensemble = np.mean(ranked_test_preds, axis=0)
    
    # Method 2: Calculate MSE-based weights
    mse_values = []
    for i, pred in enumerate(val_preds):
        mse = ((pred - y_val) ** 2).mean()
        mse_values.append(mse)
    
    # Convert MSE to weights (lower MSE = higher weight)
    if all(mse > 0 for mse in mse_values):
        weights = [1/mse for mse in mse_values]
        # Normalize weights to sum to 1
        weights = [w/sum(weights) for w in weights]
    else:
        # Equal weights if there's a problem
        weights = [1/len(mse_values)] * len(mse_values)
    
    print(f"MSE-based weights: {[round(w, 4) for w in weights]}")
    
    # Apply weights to test predictions
    weighted_pred = np.zeros(test_predictions[0].shape)
    for i, pred in enumerate(test_predictions):
        weighted_pred += weights[i] * pred
    
    # Method 3: Simple averaging
    avg_pred = np.mean(test_predictions, axis=0)
    
    # Final meta-ensemble combining all approaches
    final_ensemble = (
        0.40 * weighted_pred + 
        0.35 * avg_pred + 
        0.25 * rank_ensemble
    )
    
    return final_ensemble, test_predictions, weights

def nested_cross_validation(X, y, feature_set, run_dir, outer_cv=5):
    """Perform nested cross-validation with distribution matching"""
    print(f"\nPerforming nested CV with {feature_set} feature set...")
    
    # Get features for this set
    feature_sets = get_feature_sets(X)
    X_selected = X[feature_sets[feature_set]]
    
    # Initialize outer CV
    outer_cv_splits = KFold(n_splits=outer_cv, shuffle=True, random_state=42)
    
    # Store scores from each fold
    fold_scores = []
    fold_models = []
    
    # For storing out-of-fold predictions
    oof_preds = np.zeros(len(X))
    
    # Loop through outer CV folds
    for fold, (train_idx, test_idx) in enumerate(outer_cv_splits.split(X_selected)):
        print(f"Fold {fold+1}/{outer_cv}")
        
        # Split data for this fold
        X_train_fold, X_test_fold = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        
        # Further split training data for ensemble validation
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train_fold, y_train_fold, test_size=0.2, random_state=fold)
        
        # Train models
        models, val_preds = train_advanced_models(X_train_sub, y_train_sub, X_val, y_val, feature_set)
        
        # Create ensemble prediction for the test fold
        ensemble_pred, test_fold_preds, weights = create_advanced_ensemble(
            models, val_preds, y_val, X_test_fold)
        
        # Calibrate distribution
        calibrated_pred = calibrate_rainfall_distribution(y_train_fold, ensemble_pred)
        
        # Evaluate MSE
        fold_mse = ((calibrated_pred - y_test_fold) ** 2).mean()
        fold_scores.append(fold_mse)
        
        # Store out-of-fold predictions
        oof_preds[test_idx] = calibrated_pred
        
        print(f"Fold {fold+1} MSE: {fold_mse:.6f}")
        
        # Save fold models
        fold_dir = os.path.join(run_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        
        for i, (model_name, model) in enumerate(models):
            model_path = os.path.join(fold_dir, f'{model_name}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        fold_models.append((fold, models, weights))
    
    # Calculate overall score
    overall_mse = ((oof_preds - y) ** 2).mean()
    print(f"Overall nested CV MSE: {overall_mse:.6f}")
    
    # Save out-of-fold predictions
    oof_df = pd.DataFrame({
        'true_rainfall': y,
        'prediction': oof_preds
    })
    oof_path = os.path.join(run_dir, f'{feature_set}_oof_predictions.csv')
    oof_df.to_csv(oof_path, index=False)
    
    return overall_mse, fold_scores, fold_models

def final_training(X, y, X_test, feature_set, run_dir):
    """Train final models on full dataset"""
    print(f"\nTraining final models on full dataset using {feature_set} features...")
    
    # Get features for this set
    feature_sets = get_feature_sets(X)
    X_selected = X[feature_sets[feature_set]]
    X_test_selected = X_test[feature_sets[feature_set]]
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_selected, y, test_size=0.2, random_state=42)
    
    # Train models
    models, val_preds = train_advanced_models(X_train, y_train, X_val, y_val, feature_set)
    
    # Create ensemble prediction
    ensemble_pred, test_preds, weights = create_advanced_ensemble(
        models, val_preds, y_val, X_test_selected)
    
    # Calibrate distribution
    calibrated_pred = calibrate_rainfall_distribution(y, ensemble_pred)
    
    # Save the final models
    final_model_dir = os.path.join(run_dir, 'final_models')
    os.makedirs(final_model_dir, exist_ok=True)
    
    for model_name, model in models:
        model_path = os.path.join(final_model_dir, f'{model_name}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    # Also save weights for future inference
    weights_path = os.path.join(final_model_dir, 'ensemble_weights.csv')
    pd.DataFrame({
        'model': [m[0] for m in models],
        'weight': weights
    }).to_csv(weights_path, index=False)
    
    return calibrated_pred, models, weights

def generate_submission(pred, test_df, feature_set, run_dir):
    """Generate submission file"""
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'id': test_df['id'],
        'rainfall': pred
    })
    
    # Format with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = os.path.join('data/candidates', f'super_model_{feature_set}_{timestamp}.csv')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    
    # Save
    submission.to_csv(submission_path, index=False)
    
    # Calculate and print stats
    print("\nSubmission statistics:")
    print(f"Min: {pred.min():.6f}")
    print(f"Max: {pred.max():.6f}")
    print(f"Mean: {pred.mean():.6f}")
    print(f"Zeros: {(pred == 0).sum()} ({(pred == 0).mean()*100:.2f}%)")
    
    print(f"Submission saved to {submission_path}")
    return submission_path

def main():
    """Main function to run the super model pipeline - STEP 5 Integration"""
    print("Starting super model pipeline for breaking the 0.86+ barrier...")
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('/home/USER/Projects/RAIN/data/train.csv')
    test_df = pd.read_csv('/home/USER/Projects/RAIN/data/test.csv')
    
    # Add meteorological features
    print("Adding meteorological features...")
    train_df = add_meteorological_features(train_df)
    test_df = add_meteorological_features(test_df)
    
    # Prepare features and target
    y = train_df['rainfall']
    X = train_df.drop(['id', 'rainfall'], axis=1, errors='ignore')
    X_test = test_df.drop(['id'], axis=1, errors='ignore')
    
    # Align X_test columns with X
    missing_cols = set(X.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0  # Default value
    
    # Ensure same column order
    X_test = X_test[X.columns]
    
    # Create run directory
    run_dir = create_run_dir()
    print(f"Results will be saved to {run_dir}")
    
    # Get feature sets
    feature_sets = get_feature_sets(X)
    print(f"Available feature sets: {list(feature_sets.keys())}")
    
    # Dictionary to store results
    results = {}
    best_score = float('inf')
    best_feature_set = None
    
    # For each feature set, train models
    for feature_set in feature_sets.keys():
        print(f"\n{'='*50}")
        print(f"Processing feature set: {feature_set}")
        print(f"Number of features: {len(feature_sets[feature_set])}")
        print(f"{'='*50}")
        
        # Perform nested cross-validation
        cv_mse, fold_scores, fold_models = nested_cross_validation(
            X, y, feature_set, run_dir, outer_cv=3)  # Using 3 folds for speed
        
        # Track best feature set
        if cv_mse < best_score:
            best_score = cv_mse
            best_feature_set = feature_set
        
        # Train final model and generate predictions
        final_pred, final_models, weights = final_training(
            X, y, X_test, feature_set, run_dir)
        
        # Generate submission
        submission_path = generate_submission(
            final_pred, test_df, feature_set, run_dir)
        
        # Store results
        results[feature_set] = {
            'cv_mse': cv_mse,
            'fold_scores': fold_scores,
            'submission_path': submission_path
        }
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY OF RESULTS")
    print("="*50)
    
    for feature_set, result in results.items():
        print(f"\nFeature set: {feature_set}")
        print(f"Number of features: {len(feature_sets[feature_set])}")
        print(f"Nested CV MSE: {result['cv_mse']:.6f}")
        print(f"Fold MSEs: {[round(score, 6) for score in result['fold_scores']]}")
        print(f"Submission: {os.path.basename(result['submission_path'])}")
    
    print(f"\nBest feature set: {best_feature_set} with CV MSE {best_score:.6f}")
    print(f"Recommended submission: {os.path.basename(results[best_feature_set]['submission_path'])}")

if __name__ == "__main__":
    main()