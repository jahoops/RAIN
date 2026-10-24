import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from data_loader import load_data, preprocess_data

def identify_rainfall_patterns(train_df: pd.DataFrame, 
                              day_column: str = 'day',
                              rainfall_column: str = 'rainfall',
                              z_threshold: float = 1.0,  # Lower threshold to ~68% confidence
                              min_segment_length: int = 5) -> Dict:
    """
    Identify statistically significant rainfall patterns based on days of year
    
    Args:
        train_df: Training dataframe containing day and rainfall data
        day_column: Name of column containing day information
        rainfall_column: Name of column containing rainfall information
        z_threshold: Z-score threshold for statistical significance
        min_segment_length: Minimum length of contiguous day segments
        
    Returns:
        Dictionary with day ranges and their classifications
    """
    if day_column not in train_df.columns or rainfall_column not in train_df.columns:
        print(f"Required columns not found in data: {day_column} or {rainfall_column}")
        return {}
    
    print(f"Analyzing rainfall patterns by day...")
    
    # Group by day and calculate statistics
    daily_stats = train_df.groupby(day_column)[rainfall_column].agg(
        ['mean', 'std', 'count']).reset_index()
    
    # Calculate global average rainfall
    global_mean = train_df[rainfall_column].mean()
    global_std = train_df[rainfall_column].std()
    
    # Calculate z-scores for each day
    daily_stats['z_score'] = (daily_stats['mean'] - global_mean) / (global_std / np.sqrt(daily_stats['count']))
    
    # After calculating z-scores, print summary statistics to debug
    print("\nZ-score summary statistics:")
    print(f"Min z-score: {daily_stats['z_score'].min():.4f}")
    print(f"Max z-score: {daily_stats['z_score'].max():.4f}")
    print(f"Mean z-score: {daily_stats['z_score'].mean():.4f}")
    print(f"Z-score threshold: {z_threshold}")
    
    # Print days with highest and lowest z-scores
    print("\nTop 5 days with highest rainfall (z-scores):")
    top_days = daily_stats.sort_values('z_score', ascending=False).head(5)
    for _, row in top_days.iterrows():
        print(f"Day {int(row[day_column])}: Mean={row['mean']:.4f}, Z-score={row['z_score']:.4f}")
    
    print("\nTop 5 days with lowest rainfall (z-scores):")
    bottom_days = daily_stats.sort_values('z_score').head(5)
    for _, row in bottom_days.iterrows():
        print(f"Day {int(row[day_column])}: Mean={row['mean']:.4f}, Z-score={row['z_score']:.4f}")
    
    # Get probability distribution information
    rainfall_mean = train_df[rainfall_column].mean()
    rainfall_std = train_df[rainfall_column].std()
    
    print(f"\nRainfall probability information:")
    print(f"Global mean probability: {rainfall_mean:.4f}")
    print(f"Global std deviation: {rainfall_std:.4f}")
    
    # Define thresholds for probability bands
    very_low_threshold = rainfall_mean - 1.5 * rainfall_std
    low_threshold = rainfall_mean - 0.5 * rainfall_std
    high_threshold = rainfall_mean + 0.5 * rainfall_std
    very_high_threshold = rainfall_mean + 1.5 * rainfall_std
    
    print(f"Very low rainfall threshold: {very_low_threshold:.4f}")
    print(f"Low rainfall threshold: {low_threshold:.4f}")
    print(f"High rainfall threshold: {high_threshold:.4f}")
    print(f"Very high rainfall threshold: {very_high_threshold:.4f}")
    
    # Classify days based on both statistical significance and actual probability
    daily_stats['rain_level'] = 'MODERATE'
    daily_stats.loc[daily_stats['mean'] <= very_low_threshold, 'rain_level'] = 'VERY_LOW'
    daily_stats.loc[(daily_stats['mean'] > very_low_threshold) & 
                   (daily_stats['mean'] <= low_threshold), 'rain_level'] = 'LOW'
    daily_stats.loc[(daily_stats['mean'] > high_threshold) & 
                   (daily_stats['mean'] <= very_high_threshold), 'rain_level'] = 'HIGH'
    daily_stats.loc[daily_stats['mean'] > very_high_threshold, 'rain_level'] = 'VERY_HIGH'
    
    # Count days in each probability band
    rain_level_counts = daily_stats['rain_level'].value_counts()
    print("\nRainfall probability level counts:")
    for level, count in rain_level_counts.items():
        print(f"{level}: {count} days")
    
    # We'll keep the statistical significance classification as well
    daily_stats['significance'] = 'AVERAGE'
    daily_stats.loc[daily_stats['z_score'] > z_threshold, 'significance'] = 'HIGH_RAIN'
    daily_stats.loc[daily_stats['z_score'] < -z_threshold, 'significance'] = 'LOW_RAIN'
    
    # Create a combined classification that considers both probability level and significance
    daily_stats['combined_pattern'] = daily_stats['rain_level']
    # For days with high statistical significance, add that information
    daily_stats.loc[daily_stats['z_score'] > z_threshold * 1.5, 'combined_pattern'] = 'SIG_' + daily_stats['rain_level']
    daily_stats.loc[daily_stats['z_score'] < -z_threshold * 1.5, 'combined_pattern'] = 'SIG_' + daily_stats['rain_level']
    
    # Count days in each category
    pattern_counts = daily_stats['significance'].value_counts()
    print("\nStatistical significance pattern counts:")
    for pattern, count in pattern_counts.items():
        print(f"{pattern}: {count} days")
    
    # Find contiguous segments with same classification
    segments = []
    current_type = None
    start_day = None
    
    for idx, row in daily_stats.iterrows():
        day = row[day_column]
        sig = row['combined_pattern']
        
        if sig != current_type:
            # End previous segment if exists
            if current_type is not None and start_day is not None:
                segments.append({
                    'start_day': start_day,
                    'end_day': day - 1,
                    'type': current_type,
                    'length': day - start_day
                })
            
            # Start new segment
            current_type = sig
            start_day = day
    
    # Add the last segment
    if current_type is not None and start_day is not None:
        segments.append({
            'start_day': start_day,
            'end_day': daily_stats[day_column].max(),
            'type': current_type,
            'length': daily_stats[day_column].max() - start_day + 1
        })
    
    # Filter out segments that are too short
    filtered_segments = [s for s in segments if s['length'] >= min_segment_length]
    
    # Merge adjacent segments of the same type
    merged_segments = []
    current_segment = None
    
    for segment in filtered_segments:
        if current_segment is None:
            current_segment = segment.copy()
        elif segment['type'] == current_segment['type'] and segment['start_day'] == current_segment['end_day'] + 1:
            # Extend current segment
            current_segment['end_day'] = segment['end_day']
            current_segment['length'] = current_segment['end_day'] - current_segment['start_day'] + 1
        else:
            # Finish current segment and start a new one
            merged_segments.append(current_segment)
            current_segment = segment.copy()
    
    # Add the last segment
    if current_segment is not None:
        merged_segments.append(current_segment)
    
    # Create mappings from day to segment type
    day_to_segment = {}  # For significance
    day_to_level = {}    # For rain levels
    
    for segment in merged_segments:
        for day in range(segment['start_day'], segment['end_day'] + 1):
            day_to_segment[day] = segment['type']
    
    # Create rain level mapping
    for _, row in daily_stats.iterrows():
        day = row[day_column]
        day_to_level[day] = row['rain_level']
    
    # Fill in any missing days with AVERAGE/MODERATE
    all_days = set(range(1, daily_stats[day_column].max() + 1))
    
    missing_days = all_days - set(day_to_segment.keys())
    for day in missing_days:
        day_to_segment[day] = 'AVERAGE'
    
    missing_days = all_days - set(day_to_level.keys())
    for day in missing_days:
        day_to_level[day] = 'MODERATE'
    
    # Print summary of segments
    print(f"\nIdentified {len(merged_segments)} significant rainfall patterns:")
    for i, segment in enumerate(merged_segments):
        print(f"{i+1}. Days {segment['start_day']}-{segment['end_day']} ({segment['length']} days): {segment['type']}")
    
    # Create visualization of rainfall patterns
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Daily rainfall averages
    plt.subplot(3, 1, 1)
    plt.plot(daily_stats[day_column], daily_stats['mean'], 'b-', alpha=0.7)
    plt.axhline(y=global_mean, color='r', linestyle='--', alpha=0.7, label=f'Global Mean: {global_mean:.3f}')
    plt.xlabel('Day of Year')
    plt.ylabel('Average Rainfall')
    plt.title('Rainfall Averages by Day of Year')
    plt.legend()
    
    # Plot 2: Statistical significance segments
    plt.subplot(3, 1, 2)
    colors = {'HIGH_RAIN': 'green', 'AVERAGE': 'gray', 'LOW_RAIN': 'orange'}
    
    for segment in merged_segments:
        if 'SIG_' in segment['type']:  # Only highlight significant segments
            # Extract the base color from the significance part
            if 'LOW' in segment['type'].upper():
                color = 'orange'
            elif 'HIGH' in segment['type'].upper():
                color = 'green'
            else:
                color = 'gray'
                
            plt.axvspan(segment['start_day'], segment['end_day'], 
                       alpha=0.3, color=color, 
                       label=segment['type'])
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.xlabel('Day of Year')
    plt.ylabel('Significance')
    plt.title('Statistically Significant Rainfall Patterns')
    
    # Plot 3: Probability level segments
    plt.subplot(3, 1, 3)
    level_colors = {
        'VERY_LOW': 'darkred', 
        'LOW': 'salmon',
        'MODERATE': 'lightgray', 
        'HIGH': 'lightblue',
        'VERY_HIGH': 'darkblue'
    }
    
    # Plot day-by-day rainfall levels
    day_list = []
    level_list = []
    for day, level in sorted(day_to_level.items()):
        day_list.append(day)
        level_list.append(level_colors[level])
    
    plt.scatter(day_list, [1]*len(day_list), c=level_list, marker='|', s=100)
    
    # Add color legend
    for level, color in level_colors.items():
        plt.plot([], [], color=color, marker='|', linestyle='', markersize=20, label=level)
    
    plt.legend()
    plt.xlabel('Day of Year')
    plt.ylabel('Rainfall Level')
    plt.title('Rainfall Probability Levels')
    
    plt.tight_layout()
    plt.savefig('/home/USER/Projects/RAIN/plots/rainfall_patterns.png')
    print("Rainfall pattern visualization saved to '/home/USER/Projects/RAIN/plots/rainfall_patterns.png'")
    
    return {
        'segments': merged_segments,
        'day_to_segment': day_to_segment,
        'day_to_level': day_to_level,
        'daily_stats': daily_stats
    }

def encode_rainfall_patterns(df: pd.DataFrame, pattern_info: Dict, day_column: str = 'day') -> pd.DataFrame:
    """
    Encode rainfall patterns as features based on day of year
    
    Args:
        df: DataFrame to enhance with rainfall pattern features
        pattern_info: Dictionary with pattern information from identify_rainfall_patterns
        day_column: Column containing day information
        
    Returns:
        DataFrame with added rainfall pattern features
    """
    if pattern_info is None or 'day_to_segment' not in pattern_info or 'day_to_level' not in pattern_info:
        print("No pattern information available for encoding")
        return df
    
    if day_column not in df.columns:
        print(f"Day column {day_column} not found in dataframe")
        return df
    
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # Map days to their pattern and level
    day_to_segment = pattern_info['day_to_segment']
    day_to_level = pattern_info['day_to_level']
    
    # Create pattern and level columns
    result_df['rainfall_pattern'] = result_df[day_column].map(
        lambda d: day_to_segment.get(d, 'AVERAGE'))
    result_df['rainfall_level'] = result_df[day_column].map(
        lambda d: day_to_level.get(d, 'MODERATE'))
    
    # One-hot encode both types of patterns
    pattern_dummies = pd.get_dummies(result_df['rainfall_pattern'], prefix='pattern')
    level_dummies = pd.get_dummies(result_df['rainfall_level'], prefix='level')
    
    # Combine all features
    result_df = pd.concat([result_df, pattern_dummies, level_dummies], axis=1)
    
    # Drop the intermediate categorical columns
    result_df = result_df.drop(['rainfall_pattern', 'rainfall_level'], axis=1)
    
    print(f"Added {len(pattern_dummies.columns) + len(level_dummies.columns)} rainfall pattern features")
    
    return result_df

def create_seasonal_patterns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create rainfall pattern features for both train and test data
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
    
    Returns:
        Tuple of (enhanced_train_df, enhanced_test_df)
    """
    # Identify rainfall patterns from training data
    pattern_info = identify_rainfall_patterns(train_df)
    
    # Apply encoding to both dataframes
    enhanced_train_df = encode_rainfall_patterns(train_df, pattern_info)
    enhanced_test_df = encode_rainfall_patterns(test_df, pattern_info)
    
    return enhanced_train_df, enhanced_test_df

def evaluate_pattern_features(train_df: pd.DataFrame):
    """
    Evaluate the predictive power of the pattern features
    
    Args:
        train_df: Training dataframe with pattern features
    """
    if 'rainfall' not in train_df.columns:
        print("No rainfall target found in dataframe")
        return
    
    # Get pattern columns - exclude the categorical columns
    pattern_cols = [col for col in train_df.columns if col.startswith('pattern_') or col.startswith('level_')]
    
    if not pattern_cols:
        print("No pattern features found")
        return
    
    print(f"Using {len(pattern_cols)} rainfall pattern features for evaluation")
    
    # Split data for evaluation
    X = train_df[pattern_cols]
    y = train_df['rainfall']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Check if target is binary or continuous
    is_binary = len(train_df['rainfall'].unique()) <= 2
    
    # Train appropriate model type based on target
    import xgboost as xgb
    if is_binary:
        model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        print("Training a classification model (binary rainfall)")
    else:
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        print("Training a regression model (continuous rainfall)")
    
    model.fit(X_train, y_train)
    
    # Evaluate with appropriate metrics
    y_pred = model.predict(X_val)
    
    if is_binary:
        from sklearn.metrics import accuracy_score, roc_auc_score
        acc = accuracy_score(y_val, y_pred)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_prob)
            print(f"\nEvaluation of pattern features alone:")
            print(f"Accuracy: {acc:.6f}")
            print(f"AUC: {auc:.6f}")
        else:
            print(f"\nEvaluation of pattern features alone:")
            print(f"Accuracy: {acc:.6f}")
    else:
        from sklearn.metrics import mean_squared_error, r2_score
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        print(f"\nEvaluation of pattern features alone:")
        print(f"RMSE: {rmse:.6f}")
        print(f"R²: {r2:.6f}")
    
    # Plot feature importances
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(12, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.bar(range(min(20, len(indices))), importances[indices][:20])
        plt.xticks(range(min(20, len(indices))), [pattern_cols[i] for i in indices][:20], rotation=45)
        plt.title('Pattern Feature Importances')
        plt.tight_layout()
        plt.savefig('/home/USER/Projects/RAIN/plots/pattern_feature_importance.png')
        print("Feature importance plot saved to '/home/USER/Projects/RAIN/plots/pattern_feature_importance.png'")

def main():
    # Load data
    train_df, test_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # Create pattern features
    enhanced_train_df, enhanced_test_df = create_seasonal_patterns(train_df, test_df)
    
    # Evaluate pattern features
    evaluate_pattern_features(enhanced_train_df)
    
    # Save enhanced dataframes
    enhanced_train_df.to_csv('/home/USER/Projects/RAIN/data/train_with_patterns.csv', index=False)
    enhanced_test_df.to_csv('/home/USER/Projects/RAIN/data/test_with_patterns.csv', index=False)
    print("\nEnhanced dataframes saved to data directory")

if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    os.makedirs('/home/USER/Projects/RAIN/plots', exist_ok=True)
    main()