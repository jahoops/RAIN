import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_data

def analyze_seasonal_patterns(train_df, day_column='day'):
    """
    Analyze seasonal patterns in rainfall data based on day column
    
    Args:
        train_df: Training data with rainfall target
        day_column: Column containing day information
    """
    if day_column not in train_df.columns or 'rainfall' not in train_df.columns:
        print(f"Required columns not found in data: {day_column} or rainfall")
        return
    
    # Create directory for plots
    os.makedirs('/home/USER/Projects/RAIN/plots', exist_ok=True)
    
    # Normalize day values to 1-365 range if needed
    df = train_df.copy()
    max_day = df[day_column].max()
    
    if max_day > 366:
        print(f"Normalizing day values (max: {max_day}) to 1-365 range")
        df['day_of_year'] = df[day_column] % 365
        df.loc[df['day_of_year'] == 0, 'day_of_year'] = 365
    else:
        df['day_of_year'] = df[day_column]
    
    # Calculate month approximation
    df['month'] = np.ceil(df['day_of_year'] / 30.44).clip(1, 12).astype(int)
    
    # Define seasons
    conditions = [
        (df['day_of_year'] >= 60) & (df['day_of_year'] <= 151),
        (df['day_of_year'] >= 152) & (df['day_of_year'] <= 243),
        (df['day_of_year'] >= 244) & (df['day_of_year'] <= 334),
        (df['day_of_year'] >= 335) | (df['day_of_year'] <= 59)
    ]
    
    season_values = ['spring', 'summer', 'fall', 'winter']
    df['season'] = np.select(conditions, season_values, default='unknown')
    
    # Analyze rainfall by day of year
    plt.figure(figsize=(14, 7))
    
    # Smooth the daily data for better visualization
    df_grouped = df.groupby('day_of_year')['rainfall'].agg(['mean', 'median', 'count']).reset_index()
    
    # Add smoothed line using rolling average
    window_size = 15  # Adjust as needed for smoothing
    df_grouped['smooth_mean'] = df_grouped['mean'].rolling(window=window_size, center=True).mean()
    
    # Create vertical bands for seasons
    season_colors = {'winter': '#ADD8E6', 'spring': '#90EE90', 'summer': '#FFFFE0', 'fall': '#DEB887'}
    
    # Plot the seasonal regions
    for season, (start, end) in zip(['winter', 'spring', 'summer', 'fall'], 
                                   [(335, 365), (60, 151), (152, 243), (244, 334)]):
        if season == 'winter':  # Winter wraps around
            plt.axvspan(1, 59, alpha=0.2, color=season_colors[season])
            plt.axvspan(335, 365, alpha=0.2, color=season_colors[season])
        else:
            plt.axvspan(start, end, alpha=0.2, color=season_colors[season])
    
    # Plot rainfall vs day
    plt.plot(df_grouped['day_of_year'], df_grouped['smooth_mean'], 'b-', linewidth=2)
    plt.scatter(df_grouped['day_of_year'], df_grouped['mean'], s=5, alpha=0.3, color='darkblue')
    
    # Add season labels
    plt.text(30, plt.ylim()[1]*0.9, "Winter", fontsize=12)
    plt.text(105, plt.ylim()[1]*0.9, "Spring", fontsize=12)
    plt.text(197, plt.ylim()[1]*0.9, "Summer", fontsize=12)
    plt.text(290, plt.ylim()[1]*0.9, "Fall", fontsize=12)
    
    plt.title('Average Rainfall by Day of Year', fontsize=14)
    plt.xlabel('Day of Year', fontsize=12)
    plt.ylabel('Average Rainfall', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/USER/Projects/RAIN/plots/seasonal_rainfall_pattern.png')
    plt.close()
    
    # Monthly rainfall patterns
    plt.figure(figsize=(10, 6))
    monthly_rainfall = df.groupby('month')['rainfall'].mean()
    sns.barplot(x=monthly_rainfall.index, y=monthly_rainfall.values)
    plt.title('Average Rainfall by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Rainfall')
    plt.savefig('/home/USER/Projects/RAIN/plots/monthly_rainfall_pattern.png')
    plt.close()
    
    # Seasonal rainfall comparison
    plt.figure(figsize=(10, 6))
    seasonal_rainfall = df.groupby('season')['rainfall'].agg(['mean', 'median', 'count']).reset_index()
    seasonal_rainfall = seasonal_rainfall.sort_values('mean', ascending=False)
    
    sns.barplot(x='season', y='mean', data=seasonal_rainfall)
    plt.title('Average Rainfall by Season')
    plt.xlabel('Season')
    plt.ylabel('Average Rainfall')
    for i, row in enumerate(seasonal_rainfall.itertuples()):
        plt.text(i, row.mean/2, f'n={row.count}', ha='center')
    plt.savefig('/home/USER/Projects/RAIN/plots/seasonal_rainfall_comparison.png')
    plt.close()
    
    # Return dataframe with seasonal features for further use
    print("\nSeasonal pattern analysis complete. Check plots directory for visualizations.")
    return df

def main():
    # Load data
    train_df, _ = load_data()
    
    # Find suitable day column
    day_columns = [col for col in train_df.columns if 'day' in col.lower()]
    
    if not day_columns:
        print("No day-related columns found in the data")
        return
    
    day_column = day_columns[0]
    print(f"Using {day_column} for seasonal analysis")
    
    # Perform seasonal analysis
    analyze_seasonal_patterns(train_df, day_column=day_column)

if __name__ == "__main__":
    main()
