import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from data_loader import load_data, preprocess_data
import pickle

def create_seasonal_features(df, day_column='day'):
    """
    Create seasonal features from date/day information
    
    Args:
        df: DataFrame containing day information
        day_column: Column name containing day information
        
    Returns:
        DataFrame with added seasonal features
    """
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Normalize day values to 1-365 range if needed
    max_day = data[day_column].max()
    
    if max_day > 366:
        print(f"Normalizing day values (max: {max_day}) to 1-365 range")
        data['day_of_year'] = data[day_column] % 365
        data.loc[data['day_of_year'] == 0, 'day_of_year'] = 365
    else:
        data['day_of_year'] = data[day_column]
    
    # Calculate month approximation (1-12)
    data['month'] = np.ceil(data['day_of_year'] / 30.44).clip(1, 12).astype(int)
    
    # Add season as categorical (spring, summer, fall, winter)
    conditions = [
        (data['day_of_year'] >= 60) & (data['day_of_year'] <= 151),   # Spring
        (data['day_of_year'] >= 152) & (data['day_of_year'] <= 243),  # Summer
        (data['day_of_year'] >= 244) & (data['day_of_year'] <= 334),  # Fall
        (data['day_of_year'] >= 335) | (data['day_of_year'] <= 59)    # Winter
    ]
    season_values = ['spring', 'summer', 'fall', 'winter']
    data['season'] = np.select(conditions, season_values, default='unknown')
    
    # Create cyclical features for day of year to preserve continuity
    data['day_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 365)
    data['day_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365)
    
    # One-hot encode seasons
    season_dummies = pd.get_dummies(data['season'], prefix='season')
    data = pd.concat([data, season_dummies], axis=1)
    
    # Create "high_rainfall_season" feature based on observed patterns
    high_rainfall_seasons = ['spring', 'fall']  # Modify based on your actual data analysis
    data['high_rainfall_season'] = data['season'].isin(high_rainfall_seasons).astype(int)
    
    return data

def train_seasonal_model(train_df, test_df=None, target_col='rainfall'):
    """
    Train a model with emphasis on seasonal features
    
    Args:
        train_df: Training data
        test_df: Test data (optional)
        target_col: Target column name
    
    Returns:
        Trained model and feature names
    """
    print("Training seasonal rainfall model...")
    
    # Find a day-related column
    day_columns = [col for col in train_df.columns if 'day' in col.lower()]
    if not day_columns:
        raise ValueError("No day-related columns found in the data")
    
    day_column = day_columns[0]
    print(f"Using {day_column} for seasonal features")
    
    # Create seasonal features
    train_seasonal = create_seasonal_features(train_df, day_column)
    
    # Separate features and target
    y = train_seasonal[target_col]
    
    # Select features for model, drop unnecessary columns
    drop_cols = ['id', target_col, 'season'] + [col for col in train_seasonal.columns if col not in train_df.columns and col not in ['day_sin', 'day_cos', 'month', 'season_spring', 'season_summer', 'season_fall', 'season_winter', 'high_rainfall_season']]
    X = train_seasonal.drop(drop_cols, axis=1, errors='ignore')
    
    # Print feature information
    print(f"Training with {X.shape[1]} features: {X.columns.tolist()}")
    
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost model with seasonal awareness
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Fit model
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mae',
        early_stopping_rounds=20,
        verbose=True
    )
    
    # Evaluate model
    val_pred = model.predict(X_val)
    mse = np.mean((val_pred - y_val) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(val_pred - y_val))
    
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    
    # Analyze feature importance
    feature_importance = model.feature_importances_
    feature_names = X.columns
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sorted_idx = np.argsort(feature_importance)
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Seasonal Feature Importance')
    os.makedirs('models', exist_ok=True)
    plt.savefig('models/seasonal_feature_importance.png')
    plt.close()
    
    # Save model and feature names
    pickle.dump(model, open('models/seasonal_model.pkl', 'wb'))
    pd.DataFrame({'feature': feature_names}).to_csv('models/seasonal_feature_names.csv', index=False)
    
    # If test data is provided, create predictions
    if test_df is not None:
        test_seasonal = create_seasonal_features(test_df, day_column)
        test_X = test_seasonal.drop(['id', 'season'], axis=1, errors='ignore')
        
        # Ensure test data has all the features needed
        for col in X.columns:
            if col not in test_X.columns:
                test_X[col] = 0
        
        # Keep only the columns used during training
        test_X = test_X[X.columns]
        
        # Generate predictions
        test_preds = model.predict(test_X)
        
        # Create submission file
        submission = pd.DataFrame({
            'id': test_df['id'],
            'rainfall': test_preds
        })
        submission.to_csv('data/seasonal_model_submission.csv', index=False)
        print("Predictions saved to 'data/seasonal_model_submission.csv'")
    
    return model, feature_names

def main():
    # Load data
    train_df, test_df = load_data()
    
    # Preprocess data
    train_processed, test_processed = preprocess_data(train_df, test_df)
    
    # Train seasonal model
    model, features = train_seasonal_model(train_processed, test_processed)
    
    print("Seasonal model training complete")

if __name__ == "__main__":
    main()
