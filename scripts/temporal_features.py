import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def add_temporal_rainfall_features(train_df, test_df, id_col='id', target_col='rainfall'):
    """
    Add temporal features based on the insight that rain is more probable
    on days before or after a rainy day (temporal autocorrelation).
    
    Args:
        train_df: Training dataframe with id and rainfall columns
        test_df: Test dataframe with id column
        id_col: Column name for ID (default 'id')
        target_col: Column name for rainfall target (default 'rainfall')
        
    Returns:
        train_df_enhanced: Training dataframe with added temporal features
        test_df_enhanced: Test dataframe with added temporal features
    """
    print("Adding temporal rainfall autocorrelation features...")
    
    # Create copies to avoid modifying originals
    train_df_enhanced = train_df.copy()
    test_df_enhanced = test_df.copy()
    
    # Check if we can extract temporal information from IDs
    try:
        # First, check if IDs contain temporal information
        # Many datasets use sequential IDs or timestamps in IDs
        id_is_temporal = False
        
        # Check if IDs are numeric and sequential
        if pd.api.types.is_numeric_dtype(train_df[id_col]):
            id_diffs = np.diff(sorted(train_df[id_col]))
            if np.median(id_diffs) == 1:
                print("IDs appear to be sequential - will extract temporal features")
                id_is_temporal = True
        
        # If IDs contain temporal information, extract it
        if id_is_temporal:
            # Sort by ID to ensure temporal order
            train_sorted = train_df_enhanced.sort_values(id_col).reset_index(drop=True)
            
            # Create lag features for training data
            if target_col in train_sorted:
                # Previous day's rainfall
                train_sorted['rainfall_lag1'] = train_sorted[target_col].shift(1)
                
                # Two days ago rainfall
                train_sorted['rainfall_lag2'] = train_sorted[target_col].shift(2)
                
                # Next day's rainfall (only for training)
                train_sorted['rainfall_lead1'] = train_sorted[target_col].shift(-1)
                
                # Fill NaNs with mean
                mean_rainfall = train_sorted[target_col].mean()
                for col in ['rainfall_lag1', 'rainfall_lag2', 'rainfall_lead1']:
                    train_sorted[col].fillna(mean_rainfall, inplace=True)
                
                # Create rolling window features
                train_sorted['rainfall_rolling_3d_mean'] = train_sorted[target_col].rolling(window=3, min_periods=1).mean()
                train_sorted['rainfall_rolling_7d_mean'] = train_sorted[target_col].rolling(window=7, min_periods=1).mean()
                
                # Visualize autocorrelation
                plt.figure(figsize=(10, 6))
                pd.plotting.lag_plot(train_sorted[target_col])
                plt.title('Lag Plot of Rainfall')
                plt.xlabel('Rainfall(t)')
                plt.ylabel('Rainfall(t+1)')
                os.makedirs('analysis/temporal', exist_ok=True)
                plt.savefig('analysis/temporal/rainfall_lag_plot.png')
                plt.close()
                
                # Create autocorrelation plot
                plt.figure(figsize=(10, 6))
                pd.plotting.autocorrelation_plot(train_sorted[target_col])
                plt.title('Autocorrelation Plot of Rainfall')
                plt.savefig('analysis/temporal/rainfall_autocorrelation.png')
                plt.close()
                
                # Now apply similar transformations to test data
                test_sorted = test_df_enhanced.sort_values(id_col).reset_index(drop=True)
                
                # For test data, we can only use the test data itself to create features
                # Since we don't have the target, we need to initialize with NaN or a default value
                for lag_col in ['rainfall_lag1', 'rainfall_lag2', 'rainfall_lead1', 
                               'rainfall_rolling_3d_mean', 'rainfall_rolling_7d_mean']:
                    test_sorted[lag_col] = np.nan
                
                # We'll need to populate these in a more sophisticated model
                # For now, we'll use the mean from training data
                test_sorted[['rainfall_lag1', 'rainfall_lag2', 'rainfall_lead1', 
                           'rainfall_rolling_3d_mean', 'rainfall_rolling_7d_mean']] = mean_rainfall
                
                # Restore original order
                train_df_enhanced = train_sorted.sort_index()
                test_df_enhanced = test_sorted.sort_index()
                
                print(f"Added temporal autocorrelation features: {['rainfall_lag1', 'rainfall_lag2', 'rainfall_lead1', 'rainfall_rolling_3d_mean', 'rainfall_rolling_7d_mean']}")
                
            else:
                print(f"Warning: Target column '{target_col}' not found in training data")
        else:
            print("IDs don't appear to contain temporal information. Will try alternative approach.")
            
            # Alternative approach: Use KNN or other methods to estimate temporal patterns
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.preprocessing import StandardScaler
            
            # If we have target values in training
            if target_col in train_df:
                # Get only the feature columns
                feature_cols = [col for col in train_df.columns if col not in [id_col, target_col]]
                
                # Skip if we have no features
                if not feature_cols:
                    print("No feature columns found for KNN approach")
                    return train_df_enhanced, test_df_enhanced
                
                # Normalize features
                scaler = StandardScaler()
                train_features = scaler.fit_transform(train_df[feature_cols])
                test_features = scaler.transform(test_df[feature_cols])
                
                # Find nearest neighbors for each point
                knn = KNeighborsRegressor(n_neighbors=5)
                knn.fit(train_features, train_df[target_col])
                
                # For each data point, find its nearest neighbors' rainfall values
                # This approximates temporal patterns (similar weather conditions)
                train_nn_mean = np.zeros(len(train_df))
                train_nn_std = np.zeros(len(train_df))
                
                for i, features in enumerate(train_features):
                    # Find k+1 neighbors (includes the point itself)
                    distances, indices = knn.kneighbors([features], n_neighbors=6)
                    # Exclude the point itself (first neighbor)
                    neighbor_rainfalls = train_df.iloc[indices[0][1:]].reset_index(drop=True)[target_col]
                    train_nn_mean[i] = neighbor_rainfalls.mean()
                    train_nn_std[i] = neighbor_rainfalls.std()
                
                train_df_enhanced['neighbor_rainfall_mean'] = train_nn_mean
                train_df_enhanced['neighbor_rainfall_std'] = train_nn_std
                
                # For test data, predict neighbor rainfall values
                test_df_enhanced['neighbor_rainfall_mean'] = knn.predict(test_features)
                test_df_enhanced['neighbor_rainfall_std'] = 0.1  # Default value
                
                print("Added neighbor-based rainfall features as a proxy for temporal patterns")
            else:
                print("No target column available for neighbor-based approach")
    
    except Exception as e:
        print(f"Error adding temporal features: {str(e)}")
    
    return train_df_enhanced, test_df_enhanced

def analyze_temporal_patterns():
    """
    Analyze temporal patterns in the dataset to see if there's
    autocorrelation in rainfall that could be exploited.
    """
    print("Analyzing temporal patterns in rainfall data...")
    
    # Try to load training data with rainfall values
    train_path = 'data/train.csv'
    if not os.path.exists(train_path):
        print(f"Training data not found at {train_path}")
        return False
    
    try:
        train_df = pd.read_csv(train_path)
        if 'rainfall' not in train_df.columns:
            print("No rainfall column in training data")
            return False
            
        # Sort by ID (assuming ID has temporal meaning)
        train_df = train_df.sort_values('id').reset_index(drop=True)
        
        # Check for autocorrelation
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        from statsmodels.tsa.stattools import acf, pacf
        
        # Create output directory
        os.makedirs('analysis/temporal', exist_ok=True)
        
        # Calculate autocorrelation
        acf_values = acf(train_df['rainfall'], nlags=20, fft=False)
        
        # Plot autocorrelation
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(acf_values)), acf_values, width=0.3)
        plt.axhline(y=0, linestyle='-', color='black', alpha=0.3)
        
        # Add confidence intervals
        conf_interval = 1.96 / np.sqrt(len(train_df))
        plt.axhline(y=conf_interval, linestyle='--', color='grey', alpha=0.7)
        plt.axhline(y=-conf_interval, linestyle='--', color='grey', alpha=0.7)
        
        plt.title('Autocorrelation Function (ACF) for Rainfall')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.savefig('analysis/temporal/rainfall_acf.png')
        
        # Check for significant autocorrelation
        significant_lags = np.where(np.abs(acf_values) > conf_interval)[0]
        significant_lags = significant_lags[significant_lags > 0]  # Exclude lag 0
        
        if len(significant_lags) > 0:
            print(f"Found significant autocorrelation at lags: {significant_lags}")
            print("This suggests that adding temporal features could improve model performance")
            return True
        else:
            print("No significant autocorrelation detected")
            return False
            
    except Exception as e:
        print(f"Error analyzing temporal patterns: {str(e)}")
        return False

if __name__ == "__main__":
    # Analyze if there are temporal patterns in the data
    has_temporal_patterns = analyze_temporal_patterns()
    
    if has_temporal_patterns:
        print("\nRecommendation: Add temporal features to your models to capture rainfall autocorrelation")
        print("Run the following in your model building code:")
        print("from temporal_features import add_temporal_rainfall_features")
        print("train_df, test_df = add_temporal_rainfall_features(train_df, test_df)")
    else:
        print("\nNo strong temporal patterns detected. Standard features may be sufficient.")
