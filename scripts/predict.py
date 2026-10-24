import os
import pandas as pd
import numpy as np
import pickle
import argparse
from typing import List

def load_model(model_path: str):
    """Load the trained model"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def align_features(test_df: pd.DataFrame, expected_features: List[str]) -> pd.DataFrame:
    """
    Align test data features with the features expected by the model
    
    Args:
        test_df: Test dataframe
        expected_features: List of features expected by the model
    
    Returns:
        DataFrame with aligned features
    """
    test = test_df.copy()
    
    # Add missing features
    for feature in expected_features:
        if feature not in test.columns:
            if feature == 'id':
                # Add ID column as sequential numbers if missing
                test['id'] = np.arange(len(test))
                print(f"Added missing feature '{feature}' as sequential numbers")
            else:
                # Fill other missing features with zeros
                test[feature] = 0
                print(f"Added missing feature '{feature}' filled with zeros")
    
    # Ensure columns are in the same order
    test = test[expected_features]
    return test

def predict_rainfall(model_path: str, test_path: str, feature_names_path: str = None, 
                     output_path: str = 'data/predictions.csv'):
    """
    Make rainfall predictions and save to a submission file
    
    Args:
        model_path: Path to the trained model file
        test_path: Path to the test data
        feature_names_path: Path to saved feature names (optional)
        output_path: Path to save predictions
    """
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    print(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"Test data shape: {test_df.shape}")
    
    # Store test IDs for submission
    if 'id' in test_df.columns:
        test_ids = test_df['id'].copy()
    else:
        test_ids = pd.Series(np.arange(2190, 2190 + len(test_df)))
        print("No 'id' column found, using generated IDs starting from 2190")
    
    # Get model expected features
    expected_features = None
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
        print(f"Model expected features: {expected_features}")
    elif feature_names_path and os.path.exists(feature_names_path):
        expected_features = pd.read_csv(feature_names_path).iloc[:, 0].tolist()
        print(f"Loaded feature names from {feature_names_path}: {expected_features}")
    
    # Align features if we know what's expected
    if expected_features is not None:
        test_features = test_df.columns.tolist()
        print(f"Original test features: {test_features}")
        
        # Check for id column handling
        should_include_id = 'id' in expected_features
        if 'id' in test_df.columns and not should_include_id:
            test_df = test_df.drop('id', axis=1)
            print("Dropped 'id' column as it's not used by the model")
        
        # Align features
        test_df = align_features(test_df, [f for f in expected_features if f != 'id' or should_include_id])
        print(f"Test data shape after alignment: {test_df.shape}")
    
    # Make predictions
    try:
        predictions = model.predict(test_df)
        print(f"Generated {len(predictions)} predictions")
        
        # Create submission dataframe
        submission = pd.DataFrame({
            'id': test_ids,
            'rainfall': predictions
        })
        
        # Save to CSV
        submission.to_csv(output_path, index=False)
        print(f"Saved predictions to {output_path}")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print(f"Test data shape: {test_df.shape}")
        if expected_features is not None:
            print(f"Expected features: {expected_features}")
            print(f"Actual features: {test_df.columns.tolist()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make rainfall predictions')
    parser.add_argument('--model_path', type=str, default='models/rainfall_model.pkl',
                        help='Path to the trained model file')
    parser.add_argument('--test_path', type=str, default='data/test.csv',
                        help='Path to the test data file')
    parser.add_argument('--feature_names_path', type=str, default='models/feature_names.csv',
                        help='Path to saved feature names (optional)')
    parser.add_argument('--output_path', type=str, default='data/submission.csv',
                        help='Path to save predictions')
    
    args = parser.parse_args()
    predict_rainfall(args.model_path, args.test_path, args.feature_names_path, args.output_path)
