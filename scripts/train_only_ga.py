"""Simple GA training only on train.csv with temporal features"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from simple_ga_validation import SimpleTrainValidator
from robust_ga import RobustGeneticFeatureEngineering
import datetime

# Load only the train.csv file
print("Loading training data...")
train = pd.read_csv('/home/USER/Projects/RAIN/data/train.csv')
test = pd.read_csv('/home/USER/Projects/RAIN/data/test.csv')

# Function to create temporal features
def add_temporal_features(df):
    """Add temporal features to the dataframe"""
    print("Adding temporal features...")
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Check if we already have date/time columns
    date_cols = [col for col in df.columns if any(x in col.lower() for x in ['date', 'time', 'day', 'month', 'year'])]
    
    # If no explicit date columns, try to create a date column if we have numeric ID
    if 'id' in df.columns and not date_cols:
        # Assume IDs are sequential and might represent time periods
        df['temp_id'] = df['id']
        
        # Create pseudo-date features from ID - assuming they represent sequential time periods
        df['day_num'] = (df['temp_id'] % 365) + 1
        df['month_num'] = ((df['temp_id'] % 365) // 30) + 1
        df['year_num'] = 2000 + (df['temp_id'] // 365)
        
        # Create week and day of week
        df['week_num'] = (df['day_num'] // 7) + 1
        df['day_of_week'] = df['day_num'] % 7
    
    # If we have day/month info already use that
    if 'day' in df.columns and 'month' in df.columns:
        # These are actual date columns
        if 'year' in df.columns:
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        else:
            # Assume current year if no year column
            current_year = datetime.datetime.now().year
            df['date'] = pd.to_datetime(df[['month', 'day']].assign(year=current_year))
            
        # Extract more temporal features
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
    # Create cyclical features for periodic variables
    for col in ['day_of_year', 'month_num', 'month', 'day_of_week']:
        if col in df.columns:
            if col == 'day_of_year':
                max_val = 365
            elif col in ['month_num', 'month']:
                max_val = 12
            elif col == 'day_of_week':
                max_val = 7
            else:
                continue
                
            # Create sin and cos transformations
            df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
            df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    
    # Create lag features if dataset has a clear temporal sequence
    # This requires that rows are ordered by time
    if 'rainfall' in df.columns:
        # Create lags for rainfall
        for lag in [1, 2, 3, 7]:
            df[f'rainfall_lag_{lag}'] = df['rainfall'].shift(lag)
        
        # Create rolling statistics
        for window in [3, 7, 14]:
            df[f'rainfall_roll_mean_{window}'] = df['rainfall'].rolling(window=window).mean()
            df[f'rainfall_roll_max_{window}'] = df['rainfall'].rolling(window=window).max()
            df[f'rainfall_roll_std_{window}'] = df['rainfall'].rolling(window=window).std()
    
    # Fill NaN values with medians for numeric columns
    for col in df.columns:
        if df[col].dtype.kind in 'fib' and df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    print(f"Added {len(df.columns) - len(train.columns)} temporal features")
    return df

# Add temporal features
train_with_temporal = add_temporal_features(train)
test_with_temporal = add_temporal_features(test)

# Ensure test data has all columns from training data (with appropriate defaults)
print("Aligning test features with training features...")
missing_cols = set(train_with_temporal.columns) - set(test_with_temporal.columns) - {'rainfall'}

for col in missing_cols:
    # Add missing columns to test data with appropriate defaults
    if 'lag' in col or 'roll' in col:
        # For lag/rolling features based on rainfall, use zeros
        test_with_temporal[col] = 0
    else:
        # For other features, use median from training
        if col in train_with_temporal.columns:
            test_with_temporal[col] = train_with_temporal[col].median()
        else:
            test_with_temporal[col] = 0

print(f"Added {len(missing_cols)} missing columns to test data")

# Basic preprocessing
X = train_with_temporal.drop(['id', 'rainfall'], axis=1)
y = train_with_temporal['rainfall']

# Create a simple validator
validator = SimpleTrainValidator(X, y)

# Test baseline
rf = RandomForestRegressor(n_estimators=100, random_state=42)
scores = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error')
mse = -scores.mean()
print(f"Baseline MSE with all features + temporal: {mse:.6f}")

# Initialize and run the GA
ga = RobustGeneticFeatureEngineering(
    X_train=X,
    y_train=y,
    X_test=test_with_temporal.drop(['id'], axis=1),  # Use test with temporal features
    population_size=30,
    generations=10,
    mutation_rate=0.3,
    crossover_rate=0.7
)

# Override validator
ga.validator = validator

# Run evolution
best_chromosome, best_fitness = ga.evolve()

# Create submission
if best_fitness > 0.1:
    submission = ga.create_submission(test_with_temporal)
    submission_path = '/home/USER/Projects/RAIN/data/candidates/ga_temporal_features.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}")