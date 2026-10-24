import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def calculate_daily_rainfall_pattern(train_df, day_column='day', target_column='rainfall'):
    """
    Calculate historical rainfall probability pattern for each day of the year
    
    Args:
        train_df: Training dataframe containing day and rainfall data
        day_column: Column containing day of year information
        target_column: Column containing rainfall target
        
    Returns:
        DataFrame with daily rainfall probabilities
    """
    # Ensure day is within 1-365 range
    train_df = train_df.copy()
    train_df['day_of_year'] = train_df[day_column] % 365
    train_df.loc[train_df['day_of_year'] == 0, 'day_of_year'] = 365
    
    # Calculate average rainfall probability for each day of year
    daily_pattern = train_df.groupby('day_of_year')[target_column].mean().reset_index()
    
    # Apply smoothing with rolling average to handle sparse data
    window_size = 15  # Adjust this based on data density
    daily_pattern['smoothed_rainfall'] = daily_pattern[target_column].rolling(
        window=window_size, center=True, min_periods=1).mean()
    
    # Handle edge cases with circular smoothing (end of year connects to start of year)
    # This ensures we don't have edge effects at the year boundaries
    indices = np.arange(-window_size//2, window_size//2+1)
    for i in range(1, window_size//2+1):
        day_val = daily_pattern.loc[daily_pattern['day_of_year'] == i, target_column].values[0]
        wrap_idx = 365 - window_size//2 + i
        if wrap_idx <= 365:
            daily_pattern.loc[daily_pattern['day_of_year'] == wrap_idx, target_column] = day_val
    
    # Calculate the overall mean for normalization
    overall_mean = daily_pattern['smoothed_rainfall'].mean()
    
    # Calculate adjustment factor (how much higher/lower than average is each day)
    daily_pattern['adjustment_factor'] = daily_pattern['smoothed_rainfall'] / overall_mean
    
    # Plot the seasonal pattern
    plt.figure(figsize=(15, 6))
    plt.plot(daily_pattern['day_of_year'], daily_pattern['adjustment_factor'], color='blue')
    plt.axhline(y=1.0, color='red', linestyle='--')
    plt.title('Seasonal Rainfall Pattern (Adjustment Factor by Day of Year)')
    plt.xlabel('Day of Year')
    plt.ylabel('Rainfall Adjustment Factor')
    plt.grid(True, alpha=0.3)
    
    os.makedirs('models', exist_ok=True)
    plt.savefig('models/seasonal_rainfall_pattern.png')
    plt.close()
    
    return daily_pattern

def apply_seasonal_adjustment(predictions, days, seasonal_pattern):
    """
    Adjust predictions based on seasonal rainfall patterns
    
    Args:
        predictions: Model's original predictions
        days: Day values corresponding to each prediction
        seasonal_pattern: DataFrame with daily adjustment factors
        
    Returns:
        Seasonally adjusted predictions
    """
    # Normalize days to 1-365 range
    days_of_year = days % 365
    days_of_year[days_of_year == 0] = 365
    
    # Create a lookup dictionary for quick access to adjustment factors
    adjustment_dict = dict(zip(
        seasonal_pattern['day_of_year'], 
        seasonal_pattern['adjustment_factor']
    ))
    
    # Get adjustment factor for each prediction
    adjustment_factors = np.array([adjustment_dict.get(day, 1.0) for day in days_of_year])
    
    # Apply adjustment (multiply predictions by adjustment factor)
    adjusted_predictions = predictions * adjustment_factors
    
    # Cap predictions at 1.0 if they exceed it
    adjusted_predictions = np.minimum(adjusted_predictions, 1.0)
    
    return adjusted_predictions
