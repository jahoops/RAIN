import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from data_loader import load_data, preprocess_data
from typing import Tuple, List, Optional

def analyze_data(train_df, test_df):
    """Analyze the dataset and visualize key patterns"""
    print("Analyzing data characteristics...")
    
    # Check for target variable distribution
    if 'rainfall' in train_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(train_df['rainfall'], kde=True)
        plt.title('Rainfall Distribution')
        plt.savefig('/home/USER/Projects/RAIN/plots/rainfall_distribution.png')
        plt.close()
        
        # Check binary vs continuous
        unique_vals = train_df['rainfall'].nunique()
        print(f"Unique rainfall values: {unique_vals}")
        if unique_vals <= 2:
            print("Target appears to be binary (rain/no rain)")
        else:
            print("Target appears to be continuous (rainfall amount)")
    
    # Check feature correlations
    if 'rainfall' in train_df.columns:
        features = train_df.select_dtypes(include=['number']).drop('rainfall', axis=1, errors='ignore')
        
        # Calculate correlations with target
        corr_with_target = pd.DataFrame(features.corrwith(train_df['rainfall'])
                                      .sort_values(ascending=False)
                                      .rename('correlation'))
        
        print("\nTop features correlated with rainfall:")
        print(corr_with_target.head(10))
        
        # Plot correlation heatmap for top features
        top_features = corr_with_target.head(15).index.tolist()
        if len(top_features) > 0:
            plt.figure(figsize=(12, 10))
            corr_matrix = train_df[top_features + ['rainfall']].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Feature Correlation with Rainfall')
            plt.savefig('/home/USER/Projects/RAIN/plots/feature_correlation.png')
            plt.close()
    
    # Check if day or date features exist
    time_cols = [col for col in train_df.columns if any(x in col.lower() for x in ['day', 'date', 'time', 'year', 'month'])]
    if time_cols:
        print(f"\nTime-related columns found: {time_cols}")
        for col in time_cols:
            if train_df[col].dtype == 'object':
                print(f"Converting {col} to datetime...")
                try:
                    train_df[col] = pd.to_datetime(train_df[col])
                    if col in test_df.columns:
                        test_df[col] = pd.to_datetime(test_df[col])
                except:
                    print(f"Could not convert {col} to datetime, will treat as categorical")
            
            # Visualize time patterns if numeric
            if pd.api.types.is_numeric_dtype(train_df[col]) and 'rainfall' in train_df.columns:
                plt.figure(figsize=(12, 6))
                sns.lineplot(x=col, y='rainfall', data=train_df.sample(min(5000, len(train_df))))
                plt.title(f'Rainfall vs {col}')
                plt.savefig(f'/home/USER/Projects/RAIN/plots/rainfall_vs_{col}.png')
                plt.close()
    
    # Create directory for plots if it doesn't exist
    os.makedirs('/home/USER/Projects/RAIN/plots', exist_ok=True)
    
    return time_cols

def create_seasonal_features(df, day_column='day'):
    """
    Convert day values into seasonal features.
    Assumes day values range from 1-365/366 for a year.
    """
    if day_column not in df.columns:
        print(f"Warning: {day_column} column not found in data")
        return df
    
    # Create copy of dataframe
    enhanced_df = df.copy()
    
    print(f"Creating seasonal features from {day_column}...")
    
    # Calculate day of year (normalize to 1-365 range if needed)
    day_values = enhanced_df[day_column].values
    max_day = day_values.max()
    
    if max_day > 366:
        # If days exceed a year, consider them as absolute days from some reference
        print(f"Warning: Maximum day value ({max_day}) exceeds 366, applying modulo 365")
        enhanced_df['day_of_year'] = enhanced_df[day_column] % 365
        enhanced_df.loc[enhanced_df['day_of_year'] == 0, 'day_of_year'] = 365
    else:
        # Days already in 1-366 range
        enhanced_df['day_of_year'] = enhanced_df[day_column]
    
    # Define seasons (Northern Hemisphere)
    # Spring: day 60-151 (Mar-May)
    # Summer: day 152-243 (Jun-Aug)
    # Fall: day 244-334 (Sep-Nov)
    # Winter: day 335-59 (Dec-Feb)
    
    conditions = [
        (enhanced_df['day_of_year'] >= 60) & (enhanced_df['day_of_year'] <= 151),
        (enhanced_df['day_of_year'] >= 152) & (enhanced_df['day_of_year'] <= 243),
        (enhanced_df['day_of_year'] >= 244) & (enhanced_df['day_of_year'] <= 334),
        (enhanced_df['day_of_year'] >= 335) | (enhanced_df['day_of_year'] <= 59)
    ]
    
    season_values = ['spring', 'summer', 'fall', 'winter']
    enhanced_df['season'] = np.select(conditions, season_values, default='unknown')
    
    # Month approximation (1-12)
    enhanced_df['month'] = np.ceil(enhanced_df['day_of_year'] / 30.44).clip(1, 12).astype(int)
    
    # Create cyclic features for day and month to capture periodicity
    enhanced_df['day_sin'] = np.sin(2 * np.pi * enhanced_df['day_of_year'] / 365)
    enhanced_df['day_cos'] = np.cos(2 * np.pi * enhanced_df['day_of_year'] / 365)
    
    # Quarter of year (1-4)
    enhanced_df['quarter'] = np.ceil(enhanced_df['month'] / 3).astype(int)
    
    # High/Low rainfall seasons - to be defined based on data analysis
    # Dummy values initially - adjust after analyzing rainfall patterns
    enhanced_df['is_rainy_season'] = enhanced_df['season'].isin(['spring', 'fall']).astype(int)
    
    # One-hot encode seasons for models that work better with binary features
    season_dummies = pd.get_dummies(enhanced_df['season'], prefix='season')
    enhanced_df = pd.concat([enhanced_df, season_dummies], axis=1)
    
    # Print distribution of seasons
    print("\nSeason distribution:")
    print(enhanced_df['season'].value_counts())
    
    return enhanced_df

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

def engineer_features(train_df: pd.DataFrame, test_df: pd.DataFrame, strategy: str = 'all') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Engineer features for modeling based on specified strategy
    
    Args:
        train_df: Training dataframe
        test_df: Testing dataframe
        strategy: Feature engineering strategy ('all', 'minimal', 'boruta_top')
    
    Returns:
        Tuple of processed (train_df, test_df) with engineered features
    """
    # Import here to avoid circular imports
    from rainfall_pattern_encoding import create_seasonal_patterns
    
    # Copy dataframes to avoid modifying originals
    train = train_df.copy()
    test = test_df.copy()
    
    # For strategies that specifically use rainfall patterns
    if strategy == 'rainfall_patterns':
        print("Using only rainfall pattern features...")
        train, test = create_seasonal_patterns(train, test)
        
        # Keep only ID, target, and rainfall pattern features
        pattern_cols = [col for col in train.columns if col.startswith('pattern_') or col.startswith('level_')]
        keep_cols = ['id', 'rainfall'] + pattern_cols
        train_cols = [col for col in keep_cols if col in train.columns]
        test_cols = [col for col in keep_cols if col in test.columns]
        
        train = train[train_cols]
        test = test[test_cols]
        return train, test
    
    elif strategy == 'all_with_patterns':
        print("Using all features plus rainfall patterns...")
        # First apply standard feature engineering
        train, test = create_regular_features(train, test)
        
        # Then add rainfall patterns
        train_with_patterns, test_with_patterns = create_seasonal_patterns(train_df, test_df)
        
        # Extract just the pattern columns
        pattern_cols = [col for col in train_with_patterns.columns if col.startswith('pattern_') or col.startswith('level_')]
        
        # Add pattern columns to the regular feature set
        for col in pattern_cols:
            train[col] = train_with_patterns[col]
            test[col] = test_with_patterns[col]
        
        return train, test
        
    # For other strategies, use existing code
    # ... existing code for 'all', 'minimal', and 'boruta_top' strategies ...
    
    return train, test

# Helper function for extracting regular features without rainfall patterns
def create_regular_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Copy the original function body for creating standard features
    train = train_df.copy()
    test = test_df.copy()
    
    # Basic feature engineering code goes here
    # ... (copy from the existing engineer_features function)
    
    return train, test

def create_lagged_features(df: pd.DataFrame, 
                           features: List[str], 
                           time_col: str,
                           group_col: Optional[str] = None,
                           lags: List[int] = [1, 3, 7]) -> pd.DataFrame:
    """
    Create lagged features for time series data
    
    Args:
        df: Input dataframe
        features: List of features to create lags for
        time_col: Column name containing time information
        group_col: Column name for grouping (e.g., station_id)
        lags: List of lag periods to create
    
    Returns:
        DataFrame with added lagged features
    """
    result = df.copy()
    result = result.sort_values(by=[group_col, time_col] if group_col else time_col)
    
    for feature in features:
        if feature not in result.columns:
            continue
            
        for lag in lags:
            if group_col:
                result[f'{feature}_lag_{lag}'] = result.groupby(group_col)[feature].shift(lag)
            else:
                result[f'{feature}_lag_{lag}'] = result[feature].shift(lag)
    
    # Fill NaN values created by lagging
    for col in result.columns:
        if '_lag_' in col:
            result[col].fillna(result[col].median(), inplace=True)
    
    return result

def main():
    # Create necessary directories
    os.makedirs('/home/USER/Projects/RAIN/plots', exist_ok=True)
    os.makedirs('/home/USER/Projects/RAIN/models', exist_ok=True)
    
    # Load data
    train_df, test_df = load_data()
    
    # Preprocess data
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # Analyze data
    analyze_data(train_df, test_df)
    
    # Engineer features
    X_train, X_test = engineer_features(train_df, test_df)
    
    # Save processed data
    print("\nSaving processed data...")
    X_train.to_csv('/home/USER/Projects/RAIN/data/X_train_processed.csv', index=False)
    
    if 'rainfall' in train_df.columns:
        pd.DataFrame({'target': train_df['rainfall']}).to_csv('/home/USER/Projects/RAIN/data/y_train_processed.csv', index=False)
    
    X_test.to_csv('/home/USER/Projects/RAIN/data/X_test_processed.csv', index=False)
    
    print("Feature engineering complete!")

if __name__ == "__main__":
    main()
