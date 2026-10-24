import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def check_data():
    """
    Check the data files and visualize basic statistics
    """
    # Check if data directory exists
    data_dir = 'data'
    if not os.path.exists(data_dir):
        print(f"Data directory does not exist: {data_dir}")
        return
    
    # Check if data files exist
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    if not os.path.exists(train_path):
        print(f"Training data file does not exist: {train_path}")
        return
        
    if not os.path.exists(test_path):
        print(f"Test data file does not exist: {test_path}")
        return
    
    # Load data
    print(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    
    # Print basic information
    print("\nTraining data info:")
    print(f"Shape: {train_df.shape}")
    print(f"Columns: {train_df.columns.tolist()}")
    print("\nData types:")
    print(train_df.dtypes)
    
    print("\nTest data info:")
    print(f"Shape: {test_df.shape}")
    print(f"Columns: {test_df.columns.tolist()}")
    
    # Check for missing values
    print("\nMissing values in training data:")
    print(train_df.isnull().sum())
    
    print("\nMissing values in test data:")
    print(test_df.isnull().sum())
    
    # Check target distribution if available
    if 'rainfall' in train_df.columns:
        print("\nTarget variable distribution:")
        print(train_df['rainfall'].value_counts())
        print(f"Target mean: {train_df['rainfall'].mean()}")
        
        # Plot target distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(train_df['rainfall'], kde=True)
        plt.title('Distribution of Rainfall Target')
        plt.savefig('rainfall_distribution.png')
        print("Saved target distribution plot to rainfall_distribution.png")
    else:
        print("\nWARNING: No 'rainfall' target column found in training data")
    
    # Check for duplicate rows
    print(f"\nDuplicate rows in training data: {train_df.duplicated().sum()}")
    print(f"Duplicate rows in test data: {test_df.duplicated().sum()}")
    
    # Summary statistics
    print("\nTraining data summary statistics:")
    print(train_df.describe())
    
    # Check first few rows
    print("\nFirst 5 rows of training data:")
    print(train_df.head())
    
    print("\nFirst 5 rows of test data:")
    print(test_df.head())

if __name__ == "__main__":
    check_data()
