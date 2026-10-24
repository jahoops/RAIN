import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
from typing import List, Dict, Tuple, Any, Union

from data_loader import load_data, preprocess_data
from feature_engineering import engineer_features

def create_run_dir() -> str:
    """Create a new directory for storing experiment results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f'models/robust_model_{timestamp}'
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def get_feature_sets(X: pd.DataFrame) -> Dict[str, List[str]]:
    """Create different feature sets for model training"""
    feature_sets = {}
    
    # All features
    feature_sets['all'] = X.columns.tolist()
    
    # Minimal features - the top most important weather predictors
    minimal_features = ['cloud', 'humidity', 'windspeed', 'dewpoint', 'temparature']
    minimal_features = [col for col in minimal_features if col in X.columns]
    feature_sets['minimal'] = minimal_features
    
    # Try to load Boruta selected features if they exist
    boruta_path = 'models/feature_selection/boruta_selected_features.csv'
    if os.path.exists(boruta_path):
        boruta_features = pd.read_csv(boruta_path)['feature'].tolist()
        boruta_features = [col for col in boruta_features if col in X.columns]
        feature_sets['boruta'] = boruta_features
    
    return feature_sets

def nested_cross_validation(X: pd.DataFrame, y: pd.Series, 
                           feature_set: str, 
                           run_dir: str, 
                           outer_cv=5, 
                           inner_cv=3) -> Tuple[float, List[float]]:
    """
    Implement nested cross-validation for more realistic performance estimation
    
    Args:
        X: Feature dataframe
        y: Target series
        feature_set: Name of feature set to use
        run_dir: Directory to save results
        outer_cv: Number of outer CV folds
        inner_cv: Number of inner CV folds
    
    Returns:
        Tuple of (mean_score, fold_scores)
    """
    # Get features for this set
    feature_sets = get_feature_sets(X)
    X_selected = X[feature_sets[feature_set]]
    
    # Initialize outer CV
    outer_cv_splits = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42)
    
    # Store scores from each fold
    fold_scores = []
    fold_models = []
    fold_best_params = []
    
    # For storing out-of-fold predictions
    oof_preds = np.zeros(len(X))
    
    # Loop through outer CV folds
    for fold, (train_idx, test_idx) in enumerate(outer_cv_splits.split(X_selected, y)):
        print(f"Outer fold {fold+1}/{outer_cv}")
        
        # Split data for this fold
        X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Find best hyperparameters using inner CV
        study = perform_bayesian_optimization(X_train, y_train, inner_cv)
        best_params = study.best_params
        fold_best_params.append(best_params)
        
        print(f"Best params for fold {fold+1}: {best_params}")
        
        # Train final model with best parameters on entire fold's training data
        model = get_optimized_model(best_params)
        model.fit(X_train, y_train)
        fold_models.append(model)
        
        # Evaluate on the test fold
        y_pred = model.predict_proba(X_test)[:, 1]
        fold_score = roc_auc_score(y_test, y_pred)
        fold_scores.append(fold_score)
        
        # Store out-of-fold predictions
        oof_preds[test_idx] = y_pred
        
        print(f"Fold {fold+1} score: {fold_score:.6f}")
        
        # Save fold model
        model_path = os.path.join(run_dir, f'{feature_set}_fold_{fold}_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    # Calculate overall score
    overall_score = roc_auc_score(y, oof_preds)
    print(f"Overall nested CV score: {overall_score:.6f}")
    
    # Save out-of-fold predictions
    oof_df = pd.DataFrame({
        'true_value': y,
        'prediction': oof_preds
    })
    oof_df.to_csv(os.path.join(run_dir, f'{feature_set}_oof_predictions.csv'), index=False)
    
    # Save best parameters for each fold
    pd.DataFrame(fold_best_params).to_csv(
        os.path.join(run_dir, f'{feature_set}_best_params.csv'), index=False)
    
    return overall_score, fold_scores

def get_param_space() -> Dict:
    """Define hyperparameter search space for Bayesian optimization"""
    return {
        'model_type': ['xgboost', 'lightgbm', 'catboost', 'random_forest'],
        'learning_rate': (0.005, 0.2),  # For gradient boosting models
        'n_estimators': (100, 1000),
        'max_depth': (3, 10),
        'subsample': (0.5, 1.0),
        'colsample': (0.5, 1.0),
        'min_child_weight': (1, 20),
        'reg_alpha': (0, 10),  # L1 regularization
        'reg_lambda': (0, 10),  # L2 regularization
    }

def get_optimized_model(params: Dict) -> Any:
    """
    Create a model with the optimized parameters
    
    Args:
        params: Dictionary of optimized parameters
    
    Returns:
        Trained model instance
    """
    model_type = params['model_type']
    
    if model_type == 'xgboost':
        return xgb.XGBClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample'],
            min_child_weight=params['min_child_weight'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            use_label_encoder=False,
            eval_metric='auc',
            random_state=42
        )
    elif model_type == 'lightgbm':
        return lgb.LGBMClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample'],
            min_child_weight=params['min_child_weight'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            random_state=42
        )
    elif model_type == 'catboost':
        return CatBoostClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            subsample=params['subsample'],
            random_strength=params['reg_alpha'],  # Using reg_alpha as random_strength
            l2_leaf_reg=params['reg_lambda'],
            verbose=0,
            random_state=42
        )
    elif model_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_child_weight'],
            max_features=params['colsample'],
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def perform_bayesian_optimization(X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 3) -> optuna.Study:
    """
    Perform Bayesian optimization to find optimal hyperparameters
    
    Args:
        X_train: Training features
        y_train: Training target
        cv_folds: Number of cross-validation folds for hyperparameter tuning
    
    Returns:
        Optuna study object with optimization results
    """
    # Define parameter space
    param_space = get_param_space()
    
    def objective(trial):
        # Sample hyperparameters
        params = {
            'model_type': trial.suggest_categorical('model_type', param_space['model_type']),
            'n_estimators': trial.suggest_int('n_estimators', *param_space['n_estimators']),
            'max_depth': trial.suggest_int('max_depth', *param_space['max_depth']),
            'subsample': trial.suggest_float('subsample', *param_space['subsample']),
            'colsample': trial.suggest_float('colsample', *param_space['colsample']),
            'min_child_weight': trial.suggest_int('min_child_weight', *param_space['min_child_weight']),
            'reg_alpha': trial.suggest_float('reg_alpha', *param_space['reg_alpha']),
            'reg_lambda': trial.suggest_float('reg_lambda', *param_space['reg_lambda']),
        }
        
        # Add learning_rate for gradient boosting models
        if params['model_type'] in ['xgboost', 'lightgbm', 'catboost']:
            params['learning_rate'] = trial.suggest_float(
                'learning_rate', *param_space['learning_rate'], log=True)
        else:
            params['learning_rate'] = 0.1  # Dummy value for Random Forest
        
        # Create and evaluate model using cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Get model with these parameters
        model = get_optimized_model(params)
        
        # Perform cross-validation
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Train model
            model.fit(X_cv_train, y_cv_train)
            
            # Make predictions
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict_proba(X_cv_val)[:, 1]
            else:
                y_pred = model.predict(X_cv_val)
            
            # Calculate score
            score = roc_auc_score(y_cv_val, y_pred)
            scores.append(score)
        
        # Return mean CV score
        return np.mean(scores)
    
    # Create and run Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)  # Adjust n_trials as needed
    
    return study

def create_ensemble(X: pd.DataFrame, y: pd.Series, feature_set: str, run_dir: str) -> Any:
    """
    Create an ensemble of different models for better performance
    
    Args:
        X: Feature dataframe
        y: Target series
        feature_set: Name of feature set to use
        run_dir: Directory to save results
    
    Returns:
        Trained ensemble model
    """
    # Get features for this set
    feature_sets = get_feature_sets(X)
    X_selected = X[feature_sets[feature_set]]
    
    # Split data for ensemble training
    X_train, X_val, y_train, y_val = train_test_split(
        X_selected, y, test_size=0.2, random_state=42)
    
    # Get best params for different model types from previously saved results
    best_params_path = os.path.join(run_dir, f'{feature_set}_best_params.csv')
    
    # If best params exist, load them; otherwise, use default params
    if os.path.exists(best_params_path):
        all_params = pd.read_csv(best_params_path).to_dict('records')
        
        # Extract best params for each model type
        model_params = {}
        for params in all_params:
            if params['model_type'] not in model_params:
                model_params[params['model_type']] = params
    else:
        # Use default parameters (simplified)
        model_params = {
            'xgboost': {
                'model_type': 'xgboost',
                'n_estimators': 500,
                'learning_rate': 0.03,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample': 0.8,
                'min_child_weight': 1,
                'reg_alpha': 5,
                'reg_lambda': 5
            },
            'lightgbm': {
                'model_type': 'lightgbm',
                'n_estimators': 500,
                'learning_rate': 0.03,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample': 0.8,
                'min_child_weight': 1,
                'reg_alpha': 5,
                'reg_lambda': 5
            },
            'catboost': {
                'model_type': 'catboost',
                'n_estimators': 500,
                'learning_rate': 0.03,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample': 0.8,
                'min_child_weight': 1,
                'reg_alpha': 5,
                'reg_lambda': 5
            },
            'random_forest': {
                'model_type': 'random_forest',
                'n_estimators': 500,
                'max_depth': 10,
                'subsample': 0.8,
                'colsample': 0.8,
                'min_child_weight': 1,
                'reg_alpha': 0,
                'reg_lambda': 0,
                'learning_rate': 0.1  # Dummy value for RF
            }
        }
    
    # Create base models
    models = []
    for model_type, params in model_params.items():
        model = get_optimized_model(params)
        models.append((model_type, model))
    
    # Create voting ensemble
    ensemble = VotingClassifier(
        estimators=models,
        voting='soft',  # Use probability estimates
        n_jobs=-1
    )
    
    # Train ensemble
    ensemble.fit(X_train, y_train)
    
    # Evaluate ensemble
    ensemble_pred = ensemble.predict_proba(X_val)[:, 1]
    ensemble_score = roc_auc_score(y_val, ensemble_pred)
    print(f"Ensemble validation score: {ensemble_score:.6f}")
    
    # Save ensemble model
    ensemble_path = os.path.join(run_dir, f'{feature_set}_ensemble_model.pkl')
    with open(ensemble_path, 'wb') as f:
        pickle.dump(ensemble, f)
    
    return ensemble, ensemble_score

def generate_final_submission(ensemble: Any, X: pd.DataFrame, y: pd.Series, test_df: pd.DataFrame, 
                             feature_set: str, run_dir: str) -> str:
    """
    Generate final submission using the ensemble model
    
    Args:
        ensemble: Trained ensemble model
        X: Training features
        y: Training target
        test_df: Test dataframe
        feature_set: Name of feature set used
        run_dir: Directory to save results
    
    Returns:
        Path to the submission file
    """
    # Get features for this set
    feature_sets = get_feature_sets(X)
    X_test = test_df[feature_sets[feature_set]]
    
    # Make predictions
    test_preds = ensemble.predict_proba(X_test)[:, 1]
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'rainfall': test_preds
    })
    
    # Format score with timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    submission_path = os.path.join('data', f'submission_{feature_set}_ensemble_{timestamp}.csv')
    submission.to_csv(submission_path, index=False)
    
    print(f"Final submission saved to {submission_path}")
    return submission_path

def main():
    """Main function to run the robust modeling pipeline"""
    print("Starting robust modeling pipeline...")
    
    # Load and preprocess data
    train_df, test_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # Apply feature engineering with full strategy
    train_df, test_df = engineer_features(train_df, test_df, strategy='all')
    
    # Prepare features and target
    X = train_df.drop(['id', 'rainfall'], axis=1, errors='ignore')
    y = train_df['rainfall']
    
    # Create directory for this run
    run_dir = create_run_dir()
    print(f"Results will be saved to {run_dir}")
    
    # Get feature sets
    feature_sets = get_feature_sets(X)
    print(f"Available feature sets: {list(feature_sets.keys())}")
    
    # Dictionary to store results for each feature set
    results = {}
    
    # For each feature set, run the robust modeling pipeline
    for feature_set in feature_sets.keys():
        print(f"\n{'='*50}")
        print(f"Processing feature set: {feature_set}")
        print(f"{'='*50}")
        
        # Perform nested cross-validation to get realistic performance estimates
        cv_score, fold_scores = nested_cross_validation(
            X, y, feature_set, run_dir)
        
        # Create and train ensemble model
        ensemble, ensemble_score = create_ensemble(X, y, feature_set, run_dir)
        
        # Generate final submission
        submission_path = generate_final_submission(
            ensemble, X, y, test_df, feature_set, run_dir)
        
        # Store results
        results[feature_set] = {
            'cv_score': cv_score,
            'fold_scores': fold_scores,
            'ensemble_score': ensemble_score,
            'submission_path': submission_path
        }
    
    # Print summary of results
    print("\n" + "="*50)
    print("SUMMARY OF RESULTS")
    print("="*50)
    
    for feature_set, result in results.items():
        print(f"\nFeature set: {feature_set}")
        print(f"Number of features: {len(feature_sets[feature_set])}")
        print(f"Nested CV score: {result['cv_score']:.6f}")
        print(f"Individual fold scores: {[round(score, 6) for score in result['fold_scores']]}")
        print(f"Ensemble validation score: {result['ensemble_score']:.6f}")
        print(f"Submission file: {os.path.basename(result['submission_path'])}")
    
if __name__ == "__main__":
    main()
