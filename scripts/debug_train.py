import os
import pandas as pd
import numpy as np
import time
import sys
from data_loader import load_data, preprocess_data
from feature_engineering import create_features, feature_selection
from model import train_xgboost, train_lightgbm, save_model
import argparse

def debug_print(message):
    """Print debug message with timestamp"""
    print(f"[{time.time():.2f}] {message}")

def check_file_exists(file_path):
    """Check if file exists and return size"""
    if os.path.exists(file_path):
        size = os.path.getsize(file_path) / 1024  # KB
        return True, f"File exists: {file_path} (Size: {size:.2f} KB)"
    else:
        return False, f"File does not exist: {file_path}"

def main(args):
    start_time = time.time()
    debug_print("Starting debug training")
    
    # Check if data directory exists
    data_dir = args.data_dir
    debug_print(f"Checking data directory: {data_dir}")
    if not os.path.exists(data_dir):
        debug_print(f"ERROR: Data directory does not exist: {data_dir}")
        return
    
    # Check if data files exist
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    exists, msg = check_file_exists(train_path)
    debug_print(msg)
    if not exists:
        debug_print("ERROR: Training data file missing")
        return
    
    exists, msg = check_file_exists(test_path)
    debug_print(msg)
    if not exists:
        debug_print("ERROR: Test data file missing")
        return
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    debug_print(f"Set random seed: {args.seed}")
    
    # Load data
    debug_print("Loading data...")
    load_start = time.time()
    try:
        train_df, test_df = load_data(args.data_dir)
        debug_print(f"Data loaded in {time.time() - load_start:.2f}s")
        debug_print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        
        # Check if data is empty
        if train_df.empty or test_df.empty:
            debug_print("ERROR: Data is empty")
            return
            
        # Print data info
        debug_print(f"Train columns: {train_df.columns.tolist()}")
        debug_print(f"Data types:\n{train_df.dtypes}")
        
        # Check for target column
        if 'rainfall' not in train_df.columns:
            debug_print("WARNING: 'rainfall' target column not found in training data")
        else:
            target_dist = train_df['rainfall'].value_counts()
            debug_print(f"Target distribution: {target_dist}")
            
    except Exception as e:
        debug_print(f"ERROR loading data: {str(e)}")
        return
    
    # Preprocess data
    debug_print("Preprocessing data...")
    preprocess_start = time.time()
    try:
        train_processed, test_processed = preprocess_data(train_df, test_df)
        debug_print(f"Preprocessing completed in {time.time() - preprocess_start:.2f}s")
        debug_print(f"Processed train shape: {train_processed.shape}")
    except Exception as e:
        debug_print(f"ERROR preprocessing data: {str(e)}")
        return
    
    # Create features
    debug_print("Creating features...")
    feature_start = time.time()
    try:
        X_train, X_test, y_train = create_features(train_processed, test_processed)
        debug_print(f"Feature creation completed in {time.time() - feature_start:.2f}s")
        debug_print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape if y_train is not None else 'None'}")
        
        if y_train is None:
            debug_print("ERROR: Target variable (y_train) is None")
            return
            
        # Check if we have enough samples
        if X_train.shape[0] < 10:
            debug_print("WARNING: Very small training set")
    except Exception as e:
        debug_print(f"ERROR creating features: {str(e)}")
        return
    
    # Select features
    if args.select_features:
        debug_print("Selecting features...")
        select_start = time.time()
        try:
            X_train, X_test = feature_selection(X_train, X_test, y_train, n_features=args.n_features)
            debug_print(f"Feature selection completed in {time.time() - select_start:.2f}s")
            debug_print(f"Selected X_train shape: {X_train.shape}")
        except Exception as e:
            debug_print(f"ERROR selecting features: {str(e)}")
            return
    
    # Train model
    debug_print(f"Training {args.model} model...")
    training_start = time.time()
    try:
        if args.model == 'xgboost':
            # For debugging, we'll use a more verbose configuration
            model = train_xgboost(
                X_train, y_train, 
                params={
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'eta': 0.05,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 1,
                    'tree_method': 'hist',
                    'n_estimators': 100,  # Reduced for faster debugging
                    'verbosity': 2  # More verbose output
                }
            )
        elif args.model == 'lightgbm':
            model = train_lightgbm(
                X_train, y_train,
                params={
                    'objective': 'binary',
                    'metric': 'auc',
                    'learning_rate': 0.05,
                    'num_leaves': 31,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'n_estimators': 100,  # Reduced for faster debugging
                    'verbose': 2  # More verbose output
                }
            )
        else:
            debug_print(f"ERROR: Unsupported model: {args.model}")
            return
            
        debug_print(f"Model training completed in {time.time() - training_start:.2f}s")
        
        # Check if model has important attributes
        if not hasattr(model, 'feature_importances_'):
            debug_print("WARNING: Model doesn't have feature_importances_ attribute")
        
    except Exception as e:
        debug_print(f"ERROR training model: {str(e)}")
        return
    
    # Save model
    debug_print("Saving model...")
    try:
        os.makedirs('models', exist_ok=True)
        save_model(model, os.path.join('models', f"{args.model}_model.pkl"))
        pd.Series(X_train.columns).to_csv(os.path.join('models', 'feature_names.csv'), index=False)
        debug_print("Model and feature names saved")
    except Exception as e:
        debug_print(f"ERROR saving model: {str(e)}")
        return
    
    total_time = time.time() - start_time
    debug_print(f"Total execution time: {total_time:.2f}s")
    
    if total_time < 1.0:
        debug_print("WARNING: Execution was very fast (< 1 second), which suggests a potential issue")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Debug training for rainfall prediction model')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--model', type=str, default='xgboost', choices=['xgboost', 'lightgbm'],
                        help='Model type to train')
    parser.add_argument('--select_features', action='store_true', help='Whether to select features')
    parser.add_argument('--n_features', type=int, default=20, help='Number of features to select')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    main(args)
