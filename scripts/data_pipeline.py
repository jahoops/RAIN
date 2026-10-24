import pandas as pd
import numpy as np
from data_loader import load_data, preprocess_data
from feature_engineering import engineer_features
from temporal_features import add_temporal_rainfall_features

def create_complete_data_pipeline(use_temporal=True):
    """
    Creates a complete data pipeline with proper ordering of preprocessing
    and feature engineering steps.
    
    Args:
        use_temporal: Whether to add temporal features (default: True)
        
    Returns:
        train_df: Prepared training dataframe
        test_df: Prepared test dataframe
    """
    print("Creating complete data pipeline...")
    
    # Step 1: Load raw data
    print("1. Loading raw data...")
    train_df, test_df = load_data()
    
    # Step 2: Basic preprocessing
    print("2. Applying basic preprocessing...")
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # Step 3: Basic feature engineering
    print("3. Creating basic engineered features...")
    train_df, test_df = engineer_features(train_df, test_df)
    
    # Step 4: Temporal features (after basic preprocessing and feature engineering)
    if use_temporal:
        print("4. Adding temporal rainfall features...")
        train_df, test_df = add_temporal_rainfall_features(train_df, test_df)
    
    # Print final feature count
    feature_count = len([col for col in train_df.columns if col not in ['id', 'rainfall']])
    print(f"Final feature count: {feature_count}")
    
    return train_df, test_df

def pipeline_timing_example():
    """Show how to properly sequence the data pipeline stages"""
    
    # Full pipeline with proper ordering
    print("\n=== RECOMMENDED PIPELINE ORDER ===")
    print("1. Load data")
    print("2. Preprocess data (clean, handle missing values, encode categoricals)")
    print("3. Engineer standard features")
    print("4. Add temporal features (depends on clean data with basic features)")
    
    print("\nWhy this order?")
    print("- Temporal features work best with cleaned data")
    print("- Temporal analysis needs properly preprocessed values")
    print("- Some temporal features may use engineered features as input")
    
    print("\nExample code:")
    print("""
    # Import required functions
    from data_loader import load_data, preprocess_data
    from feature_engineering import engineer_features
    from temporal_features import add_temporal_rainfall_features
    
    # Step 1: Load data
    train_df, test_df = load_data()
    
    # Step 2: Preprocess data
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # Step 3: Basic feature engineering
    train_df, test_df = engineer_features(train_df, test_df)
    
    # Step 4: Add temporal features (last)
    train_df, test_df = add_temporal_rainfall_features(train_df, test_df)
    """)

if __name__ == "__main__":
    pipeline_timing_example()
    
    # Uncomment to run the full pipeline
    # train_df, test_df = create_complete_data_pipeline()
    
    # Save prepared datasets
    # train_df.to_csv('data/train_prepared.csv', index=False)
    # test_df.to_csv('data/test_prepared.csv', index=False)
    # print("Saved prepared datasets to data/train_prepared.csv and data/test_prepared.csv")
