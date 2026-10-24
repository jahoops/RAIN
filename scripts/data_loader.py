import os
import pandas as pd
import numpy as np
from typing import Tuple

def load_data(data_dir: str = 'data') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and testing data
    
    Args:
        data_dir: Directory containing the data files
    
    Returns:
        Tuple of (train_df, test_df)
    """
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    return train_df, test_df

def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess training and testing data
    
    Args:
        train_df: Training dataframe
        test_df: Testing dataframe
    
    Returns:
        Tuple of preprocessed (train_df, test_df)
    """
    # Copy dataframes to avoid modifying originals
    train = train_df.copy()
    
    # Handle the case where test_df is None
    test = test_df.copy() if test_df is not None else None
    
    # Store ID column separately if it exists
    train_ids = None
    test_ids = None
    
    if 'id' in train.columns:
        train_ids = train['id'].copy()
        train = train.drop('id', axis=1)
    
    if test is not None and 'id' in test.columns:
        test_ids = test['id'].copy()
        test = test.drop('id', axis=1)
    
    # Handle missing values
    for df in [train, test]:
        if df is not None:
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('unknown')
                else:
                    df[col] = df[col].fillna(df[col].median())
    
    # Convert date columns to datetime if they exist
    date_columns = [col for col in train.columns if 'date' in col.lower()]
    for col in date_columns:
        if col in train.columns:
            train[col] = pd.to_datetime(train[col])
        if test is not None and col in test.columns:
            test[col] = pd.to_datetime(test[col])
    
    # Add time-based features if date columns exist
    for col in date_columns:
        if col in train.columns:
            # Extract month and season information
            train[f'{col}_month'] = train[col].dt.month
            train[f'{col}_season'] = (train[col].dt.month % 12 + 3) // 3
            
            # Create cyclical encoding for month (preserves cyclical nature)
            train[f'{col}_month_sin'] = np.sin(2 * np.pi * train[col].dt.month / 12)
            train[f'{col}_month_cos'] = np.cos(2 * np.pi * train[col].dt.month / 12)
        
        if test is not None and col in test.columns:
            # Apply same transformations to test data
            test[f'{col}_month'] = test[col].dt.month
            test[f'{col}_season'] = (test[col].dt.month % 12 + 3) // 3
            test[f'{col}_month_sin'] = np.sin(2 * np.pi * test[col].dt.month / 12)
            test[f'{col}_month_cos'] = np.cos(2 * np.pi * test[col].dt.month / 12)
    
    # Re-add ID columns if needed for identification but not for modeling
    if train_ids is not None:
        train['id'] = train_ids
    
    if test_ids is not None:
        test['id'] = test_ids
    
    return train, test
