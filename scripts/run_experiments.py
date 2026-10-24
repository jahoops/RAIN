import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import pickle
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any

from data_loader import load_data, preprocess_data
from feature_engineering import engineer_features

def create_experiment_directory() -> str:
    """Create directory for experiment results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'/home/USER/Projects/RAIN/models/experiments_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories for plots, models, and submissions
    os.makedirs(f'{results_dir}/plots', exist_ok=True)
    os.makedirs(f'{results_dir}/models', exist_ok=True)
    os.makedirs(f'{results_dir}/submissions', exist_ok=True)
    
    return results_dir

def get_model_configs() -> List[Dict]:
    """Define different model configurations to test"""
    return [
        # 1. Default XGBoost with moderate regularization
        {
            'name': 'xgb_default',
            'description': 'XGBoost with default parameters',
            'model_class': xgb.XGBRegressor,
            'params': {
                'n_estimators': 500,
                'learning_rate': 0.03,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'random_state': 42
            }
        },
        
        # 2. XGBoost with heavy regularization
        {
            'name': 'xgb_high_reg',
            'description': 'XGBoost with high regularization',
            'model_class': xgb.XGBRegressor,
            'params': {
                'n_estimators': 500,
                'learning_rate': 0.03,
                'max_depth': 5,  # Shallower trees
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'reg_alpha': 5.0,  # Stronger L1 regularization
                'reg_lambda': 5.0,  # Stronger L2 regularization
                'random_state': 42
            }
        },
        
        # 3. LightGBM model
        {
            'name': 'lgb_default',
            'description': 'LightGBM with default parameters',
            'model_class': lgb.LGBMRegressor,
            'params': {
                'n_estimators': 500,
                'learning_rate': 0.03,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'random_state': 42
            }
        },
        
        # 4. CatBoost model
        {
            'name': 'catboost',
            'description': 'CatBoost with default parameters',
            'model_class': CatBoostClassifier,
            'params': {
                'n_estimators': 500,
                'learning_rate': 0.03,
                'max_depth': 6,
                'random_strength': 1,
                'l2_leaf_reg': 3,
                'verbose': 0,
                'random_state': 42
            }
        },
        
        # 5. Random Forest (simpler model)
        {
            'name': 'random_forest',
            'description': 'Random Forest classifier',
            'model_class': RandomForestClassifier,
            'params': {
                'n_estimators': 500,
                'max_depth': 10,
                'min_samples_split': 5,
                'n_jobs': -1,
                'random_state': 42
            }
        }
    ]

def create_ensemble(base_models: List[Tuple[str, Any]]) -> VotingClassifier:
    """Create an ensemble from a list of trained models"""
    return VotingClassifier(estimators=base_models, voting='soft', n_jobs=-1)

def run_experiment(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_val: pd.DataFrame, 
    y_val: pd.Series,
    test_df: pd.DataFrame,
    model_config: Dict,
    feature_strategy: str,
    results_dir: str,
    use_ensemble: bool = False
) -> Dict:
    """
    Run a single experiment with given configuration
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        test_df: Test dataframe
        model_config: Model configuration dictionary
        feature_strategy: Feature engineering strategy
        results_dir: Directory to save results
        use_ensemble: Whether to use model ensemble
    
    Returns:
        Dictionary with experiment results
    """
    print(f"\nRunning experiment: {model_config['name']} with {feature_strategy} features")
    
    # Initialize results dictionary
    results = {
        'model_name': model_config['name'],
        'feature_strategy': feature_strategy,
        'ensemble': use_ensemble
    }
    
    start_time = time.time()
    
    # Create model or ensemble
    if use_ensemble:
        print("Creating model ensemble...")
        # Get all model configs
        model_configs = get_model_configs()
        base_models = []
        
        # Create and train each base model
        for config in model_configs:
            print(f"Training base model: {config['name']}...")
            model = config['model_class'](**config['params'])
            model.fit(X_train, y_train)
            base_models.append((config['name'], model))
        
        # Create ensemble
        model = create_ensemble(base_models)
        
    else:
        # Create single model
        print(f"Creating single model: {model_config['name']}...")
        model = model_config['model_class'](**model_config['params'])
    
    # Train model
    print("Fitting model...")
    model.fit(X_train, y_train)
    
    # Make predictions on validation set
    print("Making validation predictions...")
    if hasattr(model, 'predict_proba'):
        val_preds = model.predict_proba(X_val)[:, 1]
    else:
        val_preds = model.predict(X_val)
    
    # Make predictions on test set
    print("Making test predictions...")
    if hasattr(model, 'predict_proba'):
        test_preds = model.predict_proba(test_df.drop(['id'], axis=1))[:, 1]
    else:
        test_preds = model.predict(test_df.drop(['id'], axis=1))
    
    # Calculate validation score
    val_score = roc_auc_score(y_val, val_preds)
    
    # Save model and results
    print("Saving model and results...")
    model_name = f"{feature_strategy}_{model_config['name']}{'_ensemble' if use_ensemble else ''}"
    
    # Save model
    model_path = f"{results_dir}/models/{model_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save predictions
    val_preds_df = pd.DataFrame({
        'true_value': y_val.values,
        'prediction': val_preds
    })
    val_preds_df.to_csv(f"{results_dir}/models/{model_name}_val_preds.csv", index=False)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'rainfall': test_preds
    })
    
    submission_path = f"{results_dir}/submissions/{model_name}_score_{val_score:.6f}.csv"
    submission.to_csv(submission_path, index=False)
    
    # Create feature importance plot if available
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(12, 8))
        if use_ensemble:
            # For ensemble, use average feature importance across base models
            feature_importances = np.zeros(X_train.shape[1])
            for name, m in base_models:
                if hasattr(m, 'feature_importances_'):
                    feature_importances += m.feature_importances_
            feature_importances /= len(base_models)
        else:
            feature_importances = model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(feature_importances)[::-1]
        features = X_train.columns
        
        plt.barh(range(min(20, len(indices))), feature_importances[indices][:20])
        plt.yticks(range(min(20, len(indices))), [features[i] for i in indices][:20])
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/plots/{model_name}_feature_importance.png")
        plt.close()
    
    # Store results
    execution_time = time.time() - start_time
    results.update({
        'val_score': val_score,
        'execution_time': execution_time,
        'model_path': model_path,
        'submission_path': submission_path
    })
    
    print(f"Experiment completed in {execution_time:.2f} seconds.")
    print(f"Validation AUC score: {val_score:.6f}")
    
    return results

def run_cross_validation_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    test_df: pd.DataFrame, 
    model_config: Dict,
    feature_strategy: str,
    results_dir: str,
    n_folds: int = 5
) -> Dict:
    """
    Run experiment using cross-validation
    
    Args:
        X: Training features
        y: Training labels
        test_df: Test dataframe
        model_config: Model configuration dictionary
        feature_strategy: Feature engineering strategy
        results_dir: Directory to save results
        n_folds: Number of cross-validation folds
        
    Returns:
        Dictionary with experiment results
    """
    print(f"\nRunning CV experiment: {model_config['name']} with {feature_strategy} features")
    
    # Initialize results dictionary
    results = {
        'model_name': model_config['name'],
        'feature_strategy': feature_strategy,
        'cv_folds': n_folds
    }
    
    start_time = time.time()
    
    # Initialize cross-validation
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Arrays for storing predictions
    oof_preds = np.zeros(X.shape[0])
    test_preds = np.zeros(test_df.shape[0])
    
    fold_scores = []
    
    # Run cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"Training fold {fold+1}/{n_folds}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create and train model
        model = model_config['model_class'](**model_config['params'])
        model.fit(X_train, y_train)
        
        # Make predictions
        if hasattr(model, 'predict_proba'):
            val_fold_preds = model.predict_proba(X_val)[:, 1]
            test_fold_preds = model.predict_proba(test_df.drop(['id'], axis=1))[:, 1]
        else:
            val_fold_preds = model.predict(X_val)
            test_fold_preds = model.predict(test_df.drop(['id'], axis=1))
        
        # Store out-of-fold predictions
        oof_preds[val_idx] = val_fold_preds
        
        # Accumulate test predictions (will be averaged)
        test_preds += test_fold_preds / n_folds
        
        # Calculate fold score
        fold_score = roc_auc_score(y_val, val_fold_preds)
        fold_scores.append(fold_score)
        
        print(f"Fold {fold+1} score: {fold_score:.6f}")
        
        # Save fold model
        fold_model_path = f"{results_dir}/models/{feature_strategy}_{model_config['name']}_fold{fold}.pkl"
        with open(fold_model_path, 'wb') as f:
            pickle.dump(model, f)
    
    # Calculate overall CV score
    cv_score = roc_auc_score(y, oof_preds)
    print(f"Cross-validation score: {cv_score:.6f}")
    print(f"Individual fold scores: {[round(score, 6) for score in fold_scores]}")
    
    # Save OOF predictions
    oof_df = pd.DataFrame({
        'true_value': y,
        'prediction': oof_preds
    })
    oof_df.to_csv(f"{results_dir}/models/{feature_strategy}_{model_config['name']}_oof_preds.csv", index=False)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'rainfall': test_preds
    })
    submission_path = f"{results_dir}/submissions/{feature_strategy}_{model_config['name']}_cv_score_{cv_score:.6f}.csv"
    submission.to_csv(submission_path, index=False)
    
    # Copy best submission to main data directory with score in filename
    best_submission_path = f"/home/USER/Projects/RAIN/data/submission_{feature_strategy}_{model_config['name']}_score_{cv_score:.6f}.csv"
    submission.to_csv(best_submission_path, index=False)
    
    # Store results
    execution_time = time.time() - start_time
    results.update({
        'cv_score': cv_score,
        'fold_scores': fold_scores,
        'mean_fold_score': np.mean(fold_scores),
        'std_fold_score': np.std(fold_scores),
        'execution_time': execution_time,
        'submission_path': submission_path,
        'best_submission_path': best_submission_path
    })
    
    print(f"CV experiment completed in {execution_time:.2f} seconds.")
    
    return results

def run_all_experiments():
    """Run all experiments and compare results"""
    print("Starting experiments...")
    
    # Create directory for results
    results_dir = create_experiment_directory()
    print(f"Results will be saved to: {results_dir}")
    
    # Load and preprocess data
    train_df, test_df = load_data()
    print("Data loaded successfully")
    
    train_df, test_df = preprocess_data(train_df, test_df)
    print("Data preprocessing completed")
    
    # Feature engineering strategies to test
    feature_strategies = ['all', 'minimal', 'boruta_top', 'rainfall_patterns', 'all_with_patterns']
    
    all_results = []
    
    # Run experiments for each feature strategy
    for feature_strategy in feature_strategies:
        print(f"\n{'='*80}")
        print(f"Testing feature strategy: {feature_strategy}")
        print(f"{'='*80}")
        
        # Apply feature engineering
        try:
            print(f"Applying {feature_strategy} feature engineering...")
            train, test = engineer_features(train_df, test_df, strategy=feature_strategy)
            
            # Prepare features and target
            X = train.drop(['id', 'rainfall'], axis=1, errors='ignore')
            y = train['rainfall'] 
            
            # Train-test split for simple experiments
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Get model configurations
            model_configs = get_model_configs()
            
            # 1. Run cross-validation experiments for each model
            for model_config in model_configs:
                try:
                    results = run_cross_validation_experiment(
                        X, y, test, model_config, feature_strategy, results_dir
                    )
                    all_results.append(results)
                except Exception as e:
                    print(f"Error in CV experiment with {model_config['name']}: {e}")
            
            # 2. Create and test ensemble model
            try:
                print("\nTesting ensemble model...")
                # Use a simple train-val split for ensemble
                ensemble_results = run_experiment(
                    X_train, y_train, X_val, y_val, 
                    test, model_configs[0], feature_strategy, 
                    results_dir, use_ensemble=True
                )
                all_results.append(ensemble_results)
            except Exception as e:
                print(f"Error in ensemble experiment: {e}")
            
        except Exception as e:
            print(f"Error processing {feature_strategy} features: {e}")
    
    # Summarize results
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    
    # Sort results by score (descending)
    cv_results = [r for r in all_results if 'cv_score' in r]
    if cv_results:
        cv_results.sort(key=lambda x: x['cv_score'], reverse=True)
        
        # Create results table
        results_df = pd.DataFrame([
            {
                'Model': r['model_name'],
                'Features': r['feature_strategy'],
                'CV Score': r['cv_score'],
                'Mean Fold Score': r['mean_fold_score'],
                'Std Fold Score': r['std_fold_score'],
                'Time (s)': r['execution_time']
            }
            for r in cv_results
        ])
        
        print("\nTop 5 models by CV score:")
        print(results_df.head(5).to_string(index=False))
        
        # Save results to CSV
        results_df.to_csv(f"{results_dir}/experiment_results.csv", index=False)
        print(f"\nDetailed results saved to {results_dir}/experiment_results.csv")
        
        # Create comparison chart
        plt.figure(figsize=(12, 8))
        sns.barplot(data=results_df, x='Model', y='CV Score', hue='Features')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/model_comparison.png")
        print(f"Comparison chart saved to {results_dir}/model_comparison.png")
    
    # Create instructions for submission
    best_result = None
    if cv_results:
        best_result = cv_results[0]
        print("\n" + "="*80)
        print("BEST MODEL")
        print("="*80)
        print(f"Best model: {best_result['model_name']}")
        print(f"Feature strategy: {best_result['feature_strategy']}")
        print(f"CV Score: {best_result['cv_score']:.6f}")
        print(f"Submission file: {os.path.basename(best_result['best_submission_path'])}")
        
        # Copy best submission to main data directory
        best_submission = pd.read_csv(best_result['submission_path'])
        best_submission.to_csv('/home/USER/Projects/RAIN/data/submission.csv', index=False)
        print("\nBest submission copied to '/home/USER/Projects/RAIN/data/submission.csv'")
    
    return results_dir, all_results

if __name__ == "__main__":
    results_dir, experiment_results = run_all_experiments()