import warnings
warnings.filterwarnings("ignore")

from genetic_feature_engineering import GeneticFeatureEngineering
from data_pipeline import create_complete_data_pipeline
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import time
from copy import deepcopy

# Super reliable missing values handler - no Series comparisons
def safe_fill_missing(df):
    """Fill missing values without any Series boolean comparisons"""
    df_copy = df.copy()
    
    # Process each column directly with fillna
    for col in df_copy.columns:
        if df_copy[col].dtype in [np.float64, np.int64]:
            # For numeric columns, use median
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
        else:
            # For non-numeric, use the first mode value or "unknown"
            mode_vals = df_copy[col].mode()
            fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "unknown"
            df_copy[col] = df_copy[col].fillna(fill_val)
    
    return df_copy

def run_ga_with_base_features():
    """Run a simple GA focusing only on base features to avoid errors"""
    # Load data
    print("Loading data...")
    X_train_temporal, X_test_temporal = create_complete_data_pipeline()
    y_train = X_train_temporal['rainfall']
    X_train = X_train_temporal.drop(['id', 'rainfall'], axis=1)
    X_test = X_test_temporal.drop(['id'], axis=1)
    
    print(f"Base features: {X_train.columns.tolist()}")
    
    # Test a simple model
    X_train_part, X_val_part, y_train_part, y_val_part = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )
    
    # Get baseline performance
    model_base = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42
    )
    model_base.fit(X_train_part, y_train_part)
    val_pred = model_base.predict(X_val_part)
    base_mse = mean_squared_error(y_val_part, val_pred)
    print(f"Baseline MSE: {base_mse:.6f}, Score: {1-base_mse:.6f}")
    
    # Evaluate performance with different feature subsets
    print("\nTesting different feature subsets...")
    best_score = 0
    best_features = []
    
    # Create different feature combinations
    feature_combinations = [
        ['day', 'humidity', 'cloud', 'pressure', 'temparature', 'windspeed'],
        ['humidity', 'cloud', 'pressure', 'temparature'],
        ['day', 'humidity', 'cloud'],
        ['pressure', 'temparature', 'windspeed'],
        ['temparature', 'humidity', 'windspeed'],
        ['cloud', 'humidity', 'temparature', 'windspeed']
    ]
    
    for i, features in enumerate(feature_combinations):
        model = RandomForestRegressor(
            n_estimators=100, max_depth=None, min_samples_leaf=2,
            random_state=42+i
        )
        model.fit(X_train_part[features], y_train_part)
        val_pred = model.predict(X_val_part[features])
        mse = mean_squared_error(y_val_part, val_pred)
        score = 1 - mse
        
        print(f"Subset {i+1}: {features} - Score: {score:.6f}")
        
        if score > best_score:
            best_score = score
            best_features = features
    
    print(f"\nBest feature set: {best_features}, Score: {best_score:.6f}")
    
    # Train final model
    print("\nTraining final model...")
    final_model = RandomForestRegressor(
        n_estimators=1000,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=87
    )
    
    final_model.fit(X_train[best_features], y_train)
    
    # Create submission
    predictions = final_model.predict(X_test[best_features])
    
    submission = pd.DataFrame({
        'id': X_test_temporal['id'],
        'rainfall': predictions
    })
    
    # Save submission
    submission.to_csv('/home/USER/Projects/RAIN/data/candidates/ga_optimized_features.csv', index=False)
    print("Created submission with GA-optimized feature subset")
    
    return submission, best_features, best_score

if __name__ == "__main__":
    submission, best_features, best_score = run_ga_with_base_features()
    print(f"\nFinal submission created with score: {best_score:.6f}")
    print(f"Selected features: {best_features}")