import os
import pandas as pd
import numpy as np
import pickle
import argparse

def load_model(model_path):
    """Load the trained model"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_data(data_dir='data'):
    """Load training and test data"""
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    return train_df, test_df

def generate_submission(model_path='models/xgboost_model.pkl', 
                       data_dir='data',
                       output_path='data/submission.csv'):
    """Generate submission file"""
    # Load model
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    # Load data
    print("Loading test data...")
    train_df, test_df = load_data(data_dir)
    
    # Check test data features
    print(f"Test data shape before feature alignment: {test_df.shape}")
    test_features = test_df.columns.tolist()
    print(f"Test features: {test_features}")
    
    # Get model expected features
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_.tolist()
    else:
        # For models without feature_names_in_, extract from training data
        print("Loading feature names used during training...")
        expected_features = train_df.columns.tolist()
        # Remove target and ID columns which are not features
        if 'rainfall' in expected_features:
            expected_features.remove('rainfall')
        if 'id' in expected_features:
            expected_features.remove('id')
    
    print(f"Training features: {expected_features}")
    
    # Get ID column from test data or sample submission for output
    if 'id' in test_df.columns:
        id_values = test_df['id']
    else:
        # Get IDs from sample submission file
        sample_sub = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
        id_values = sample_sub['id']
    
    # Ensure test data only includes the expected features
    for col in test_df.columns:
        if col not in expected_features and col != 'id':
            test_df = test_df.drop(col, axis=1)
    
    # Ensure all expected features exist in test data
    for col in expected_features:
        if col not in test_df.columns:
            print(f"Warning: Missing feature in test data: {col}")
    
    print(f"Test data shape after feature alignment: {test_df.shape}")
    print(f"Final test features: {test_df.columns.tolist()}")
    
    # Generate predictions
    try:
        print("Generating rainfall probability predictions...")
        if hasattr(model, 'predict_proba'):
            # For probabilistic models
            predictions = model.predict_proba(test_df[expected_features])[:, 1]
        else:
            # For regression or binary output models
            predictions = model.predict(test_df[expected_features])
        
        # Create submission dataframe
        submission = pd.DataFrame({
            'id': id_values,
            'rainfall': predictions
        })
        
        # Save submission
        submission.to_csv(output_path, index=False)
        print(f"Submission file saved to {output_path}")
        
        # Show sample of predictions
        print("\nSample predictions:")
        print(submission.head(10))
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print(f"Model expected features: {expected_features}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate submission file')
    parser.add_argument('--model_path', type=str, default='models/xgboost_model.pkl',
                        help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='data', 
                        help='Directory containing data files')
    parser.add_argument('--output_path', type=str, default='data/submission.csv',
                        help='Path to save submission file')
    
    args = parser.parse_args()
    generate_submission(args.model_path, args.data_dir, args.output_path)
