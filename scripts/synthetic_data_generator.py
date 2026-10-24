import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, weibull_min, gamma
import matplotlib.pyplot as plt
from data_loader import load_data, preprocess_data

def analyze_data_distributions():
    """Analyze the statistical distributions of the training data"""
    train_df, _ = load_data()
    train_df, _ = preprocess_data(train_df, None)
    
    # Analyze rainfall distribution
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.hist(train_df['rainfall'], bins=50)
    plt.title('Rainfall Distribution')
    
    # Extract feature correlations with rainfall
    if 'rainfall' in train_df.columns:
        correlations = train_df.corr()['rainfall'].sort_values(ascending=False)
        top_features = correlations[1:11].index.tolist()  # Top 10 features excluding rainfall itself
        
        # Plot joint distributions for top correlated features
        for i, feature in enumerate(top_features[:3]):  # Plot top 3 for clarity
            plt.subplot(2, 2, i+2)
            plt.scatter(train_df[feature], train_df['rainfall'], alpha=0.3)
            plt.title(f'{feature} vs Rainfall (corr: {correlations[feature]:.3f})')
            plt.xlabel(feature)
            plt.ylabel('Rainfall')
    
    plt.tight_layout()
    plt.savefig('/home/USER/Projects/RAIN/plots/data_distributions.png')
    
    return top_features, correlations

def bootstrap_sampling(train_df, n_samples=1000):
    """Generate synthetic data using bootstrap sampling"""
    if n_samples <= 0:
        return pd.DataFrame()
    
    # Sample with replacement
    synthetic_df = train_df.sample(n=n_samples, replace=True, random_state=42)
    synthetic_df = synthetic_df.reset_index(drop=True)
    
    # Add small random noise to prevent exact duplicates
    for col in synthetic_df.columns:
        if synthetic_df[col].dtype in ['float64', 'int64'] and col not in ['id', 'day']:
            # Add noise proportional to the standard deviation of the column
            noise_scale = synthetic_df[col].std() * 0.05
            synthetic_df[col] = synthetic_df[col] + np.random.normal(0, noise_scale, len(synthetic_df))
    
    return synthetic_df

def parametric_sampling(train_df, n_samples=1000):
    """Generate synthetic data by fitting parametric distributions"""
    if n_samples <= 0:
        return pd.DataFrame()
    
    synthetic_df = pd.DataFrame()
    
    for col in train_df.columns:
        if col in ['id']:
            # Generate new IDs
            synthetic_df[col] = np.arange(n_samples) + max(train_df[col]) + 1
            
        elif col == 'rainfall':
            # Use Gamma distribution for rainfall (often better for meteorological data)
            # First, remove zeros or add a small constant if there are zeros
            rainfall_data = train_df[col].values
            rainfall_data = rainfall_data[rainfall_data > 0]
            
            if len(rainfall_data) > 0:
                try:
                    # Try to fit gamma distribution
                    shape, loc, scale = gamma.fit(rainfall_data)
                    synthetic_df[col] = gamma.rvs(shape, loc=loc, scale=scale, size=n_samples)
                except Exception as e:
                    print(f"Warning: Gamma distribution fitting failed: {e}")
                    print("Using lognormal distribution as fallback for rainfall")
                    
                    # Use lognormal distribution as fallback (also common for rainfall)
                    # Add a small constant to avoid log(0)
                    min_val = min(rainfall_data)
                    if min_val <= 0:
                        rainfall_data = rainfall_data + 0.01  # Small offset
                    
                    # Fit lognormal
                    shape, loc, scale = weibull_min.fit(rainfall_data)
                    synthetic_df[col] = weibull_min.rvs(shape, loc=loc, scale=scale, size=n_samples)
                
                # Ensure minimum values match the original distribution
                synthetic_df.loc[synthetic_df[col] < train_df[col].min(), col] = 0
            else:
                # Fallback to normal distribution if no positive rainfall
                mean, std = train_df[col].mean(), train_df[col].std()
                synthetic_df[col] = np.random.normal(mean, std, n_samples)
                synthetic_df[col] = np.maximum(synthetic_df[col], 0)  # Ensure no negative values
                
        elif col == 'day':
            # For time series data, maintain the temporal structure
            if train_df[col].nunique() < n_samples:
                # If we need more samples than unique days, repeat and extend
                max_day = train_df[col].max()
                days_needed = n_samples - train_df[col].nunique()
                synthetic_df[col] = list(train_df[col].unique()) + list(range(max_day + 1, max_day + days_needed + 1))
                synthetic_df = synthetic_df.sample(n=n_samples, random_state=42).reset_index(drop=True)
            else:
                # Just sample from existing days
                synthetic_df[col] = np.random.choice(train_df[col].unique(), n_samples)
                
        elif train_df[col].dtype in ['float64', 'int64']:
            # Try to fit the best distribution
            # For simplicity we'll use normal distribution here, but you could test multiple distributions
            mean, std = train_df[col].mean(), train_df[col].std()
            synthetic_df[col] = np.random.normal(mean, std, n_samples)
            
        else:
            # For categorical features, sample from the original distribution
            synthetic_df[col] = np.random.choice(train_df[col].unique(), n_samples,
                                               p=train_df[col].value_counts(normalize=True))
    
    return synthetic_df

def copula_based_sampling(train_df, n_samples=1000):
    """Generate synthetic data using Gaussian copula to preserve correlations"""
    if n_samples <= 0:
        return pd.DataFrame()
    
    # Select only numeric columns
    numeric_cols = [col for col in train_df.columns if 
                   train_df[col].dtype in ['float64', 'int64'] and col not in ['id']]
    
    numeric_df = train_df[numeric_cols].copy()
    
    # Handle NaNs if any
    numeric_df = numeric_df.fillna(numeric_df.median())
    
    # Transform to normal
    transformed_data = pd.DataFrame()
    for col in numeric_cols:
        # Calculate empirical CDF
        sorted_data = np.sort(numeric_df[col])
        ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # Transform to standard normal using inverse CDF
        transformed_data[col] = norm.ppf(ecdf[np.searchsorted(sorted_data, numeric_df[col])])
    
    # Calculate correlation matrix of transformed data
    corr_matrix = transformed_data.corr().values
    
    # Generate synthetic normal data with the same correlation
    synthetic_normal = np.random.multivariate_normal(
        mean=np.zeros(len(numeric_cols)), 
        cov=corr_matrix, 
        size=n_samples
    )
    
    # Transform back to original distributions
    synthetic_df = pd.DataFrame()
    for i, col in enumerate(numeric_cols):
        # Convert back using the original column's empirical CDF
        unif_values = norm.cdf(synthetic_normal[:, i])
        synthetic_df[col] = np.quantile(numeric_df[col].values, unif_values)
    
    # Add ID column
    synthetic_df['id'] = np.arange(n_samples) + max(train_df['id']) + 1
    
    # If day column exists, handle it specially
    if 'day' in train_df.columns:
        if 'day' in numeric_cols:
            # Round to integer
            synthetic_df['day'] = np.round(synthetic_df['day']).astype(int)
        else:
            # Sample from original days
            synthetic_df['day'] = np.random.choice(train_df['day'].unique(), n_samples)
    
    return synthetic_df

def perturbation_based_sampling(train_df, target_col='rainfall', n_samples=1000):
    """Create synthetic data by perturbing original data based on feature importance"""
    if n_samples <= 0:
        return pd.DataFrame()
    
    # First, get base samples using bootstrap
    synthetic_df = bootstrap_sampling(train_df, n_samples)
    
    # If we have rainfall predictions from our model, we could use them to guide perturbation
    # For now, we'll just add more sophisticated noise
    
    # Train a simple model to understand feature relationships
    from sklearn.ensemble import RandomForestRegressor
    if target_col in train_df.columns:
        X = train_df.drop([target_col, 'id'], axis=1, errors='ignore')
        y = train_df[target_col]
        
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)
        
        # Get feature importances
        importances = model.feature_importances_
        feature_importance = {name: importance for name, importance in zip(X.columns, importances)}
        
        # Perturb features based on their importance
        for col in synthetic_df.columns:
            if col not in ['id', target_col] and col in feature_importance:
                importance = feature_importance[col]
                # More important features get less perturbation
                noise_scale = synthetic_df[col].std() * 0.1 * (1 - importance)
                synthetic_df[col] = synthetic_df[col] + np.random.normal(0, noise_scale, len(synthetic_df))
        
        # Recalculate target column based on the model
        X_synthetic = synthetic_df.drop([target_col, 'id'], axis=1, errors='ignore')
        synthetic_df[target_col] = model.predict(X_synthetic)
        
        # Add random noise to target to avoid overfitting
        noise_scale = train_df[target_col].std() * 0.05
        synthetic_df[target_col] = synthetic_df[target_col] + np.random.normal(0, noise_scale, len(synthetic_df))
    
    return synthetic_df

def weather_pattern_simulation(train_df, n_samples=1000):
    """Create synthetic data by simulating weather patterns"""
    if 'day' not in train_df.columns or 'rainfall' not in train_df.columns:
        print("Day or rainfall column missing, cannot simulate weather patterns")
        return pd.DataFrame()
    
    # Group by day to understand daily patterns
    daily_stats = train_df.groupby('day')['rainfall'].agg(['mean', 'std', 'min', 'max'])
    
    # Analyze seasonality if enough data
    if daily_stats.shape[0] >= 90:  # At least 3 months of data
        from statsmodels.tsa.seasonal import seasonal_decompose
        try:
            # Try to decompose the time series with 7-day seasonality (weekly pattern)
            decomposition = seasonal_decompose(daily_stats['mean'], model='additive', period=7)
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid
            
            # Fill NaN values in trend and residual
            trend = trend.fillna(method='bfill').fillna(method='ffill')
            residual = residual.fillna(method='bfill').fillna(method='ffill')
            
            # Generate synthetic days
            max_day = train_df['day'].max()
            synthetic_days = np.arange(max_day + 1, max_day + n_samples + 1)
            
            # Create synthetic daily means based on the seasonal pattern
            synthetic_means = []
            for day in synthetic_days:
                season_idx = (day % 7) - 1  # Map to 0-6 for the seasonal component
                if season_idx < 0:
                    season_idx = 6  # Sunday = 0, Saturday = 6
                
                # Get the seasonal component for this day
                seasonal_component = seasonal.iloc[season_idx % len(seasonal)]
                
                # For trend, use the last observed trend plus a small increment
                trend_component = trend.iloc[-1] + (day - max_day) * (trend.iloc[-1] - trend.iloc[-2])
                
                # Add random noise based on residual standard deviation
                noise = np.random.normal(0, residual.std())
                
                synthetic_means.append(trend_component + seasonal_component + noise)
        
        except Exception as e:
            print(f"Error in seasonal decomposition: {e}")
            # Fallback to simple sampling
            synthetic_means = np.random.choice(daily_stats['mean'].values, n_samples)
            synthetic_days = np.random.choice(train_df['day'].unique(), n_samples)
    else:
        # Not enough data for seasonality, use simple statistics
        synthetic_means = np.random.normal(daily_stats['mean'].mean(), 
                                          daily_stats['mean'].std(), 
                                          n_samples)
        synthetic_days = np.arange(train_df['day'].max() + 1, 
                                  train_df['day'].max() + n_samples + 1)
    
    # Create a base dataframe with day and mean rainfall
    synthetic_df = pd.DataFrame({
        'day': synthetic_days,
        'rainfall_mean': synthetic_means
    })
    
    # Now expand to create multiple samples per day
    samples_per_day = max(1, int(n_samples / len(synthetic_df)))
    expanded_df = pd.DataFrame()
    
    for _, row in synthetic_df.iterrows():
        day = row['day']
        mean_rainfall = max(0, row['rainfall_mean'])  # Ensure non-negative
        
        # Use gamma distribution for rainfall amounts (common in meteorology)
        shape = 2.0  # Shape parameter - can be tuned
        scale = mean_rainfall / shape if mean_rainfall > 0 else 0.1
        
        # Generate rainfall values for this day
        rainfall_values = gamma.rvs(shape, scale=scale, size=samples_per_day)
        
        # Create samples for this day
        day_samples = pd.DataFrame({
            'day': [day] * samples_per_day,
            'rainfall': rainfall_values
        })
        
        expanded_df = pd.concat([expanded_df, day_samples])
    
    # Limit to requested number of samples
    expanded_df = expanded_df.iloc[:n_samples].reset_index(drop=True)
    
    # Add ID column
    expanded_df['id'] = np.arange(len(expanded_df)) + max(train_df['id']) + 1 if 'id' in train_df.columns else np.arange(len(expanded_df))
    
    # Now we need to generate other features based on rainfall and day
    if len(train_df.columns) > 3:  # If we have more features than just id, day, rainfall
        # Identify the most correlated features with rainfall
        correlations = train_df.corr()['rainfall'].sort_values(ascending=False)
        top_correlated = correlations[1:6].index.tolist()  # Top 5 features excluding rainfall itself
        
        # For each top correlated feature, generate values based on correlation with rainfall
        for feature in top_correlated:
            if feature in train_df.columns:
                # Calculate correlation coefficient
                corr = correlations[feature]
                
                # Generate correlated random variable
                mean = train_df[feature].mean()
                std = train_df[feature].std()
                
                # Create feature with the right correlation to rainfall
                z1 = np.random.normal(0, 1, len(expanded_df))
                z2 = np.random.normal(0, 1, len(expanded_df))
                x = z1 * corr + z2 * np.sqrt(1 - corr**2)
                
                # Scale and shift to match original distribution
                expanded_df[feature] = mean + std * x
        
        # For remaining features, use bootstrap sampling
        for col in train_df.columns:
            if col not in expanded_df.columns and col not in ['id', 'day', 'rainfall']:
                expanded_df[col] = np.random.choice(train_df[col].values, len(expanded_df))
    
    return expanded_df

def evaluate_synthetic_data(original_df, synthetic_df, target_col='rainfall'):
    """Evaluate how well the synthetic data mimics the original data"""
    print("Evaluating synthetic data quality...")
    
    # Compare basic statistics
    orig_stats = original_df.describe()
    synth_stats = synthetic_df.describe()
    
    comparison = pd.DataFrame({
        'Original_Mean': orig_stats.loc['mean'],
        'Synthetic_Mean': synth_stats.loc['mean'],
        'Original_Std': orig_stats.loc['std'],
        'Synthetic_Std': synth_stats.loc['std'],
        'Original_Min': orig_stats.loc['min'],
        'Synthetic_Min': synth_stats.loc['min'],
        'Original_Max': orig_stats.loc['max'],
        'Synthetic_Max': synth_stats.loc['max']
    })
    
    print("Statistical comparison:")
    print(comparison)
    
    # Compare distributions visually
    plt.figure(figsize=(15, 10))
    
    # Compare target variable distribution
    plt.subplot(2, 2, 1)
    plt.hist(original_df[target_col], bins=30, alpha=0.5, label='Original')
    plt.hist(synthetic_df[target_col], bins=30, alpha=0.5, label='Synthetic')
    plt.title(f'{target_col} Distribution Comparison')
    plt.legend()
    
    # Compare correlation matrices
    plt.subplot(2, 2, 2)
    
    # Select only numeric columns present in both dataframes
    numeric_cols = [col for col in original_df.columns if 
                   original_df[col].dtype in ['float64', 'int64'] and 
                   col in synthetic_df.columns]
    
    if len(numeric_cols) > 1:  # Need at least 2 columns for correlation
        orig_corr = original_df[numeric_cols].corr()
        synth_corr = synthetic_df[numeric_cols].corr()
        
        # Compute correlation difference
        corr_diff = abs(orig_corr - synth_corr)
        
        plt.imshow(corr_diff, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Absolute Correlation Difference')
        plt.title('Correlation Matrix Difference')
        plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
        plt.yticks(range(len(numeric_cols)), numeric_cols)
    else:
        plt.text(0.5, 0.5, "Not enough numeric columns for correlation analysis", 
                 horizontalalignment='center', verticalalignment='center')
    
    # Compare joint distributions for a few key features
    if len(numeric_cols) > 2:
        feature1 = numeric_cols[1]  # First feature after target
        plt.subplot(2, 2, 3)
        plt.scatter(original_df[feature1], original_df[target_col], alpha=0.5, label='Original')
        plt.scatter(synthetic_df[feature1], synthetic_df[target_col], alpha=0.5, label='Synthetic')
        plt.xlabel(feature1)
        plt.ylabel(target_col)
        plt.title(f'{feature1} vs {target_col}')
        plt.legend()
        
        if len(numeric_cols) > 3:
            feature2 = numeric_cols[2]  # Second feature after target
            plt.subplot(2, 2, 4)
            plt.scatter(original_df[feature2], original_df[target_col], alpha=0.5, label='Original')
            plt.scatter(synthetic_df[feature2], synthetic_df[target_col], alpha=0.5, label='Synthetic')
            plt.xlabel(feature2)
            plt.ylabel(target_col)
            plt.title(f'{feature2} vs {target_col}')
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('/home/USER/Projects/RAIN/plots/synthetic_data_quality.png')
    
    # Also check predictive equivalence
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    
    if len(original_df) > 100 and len(synthetic_df) > 100:
        # Train on original, predict on synthetic
        X_orig = original_df.drop([target_col, 'id'], axis=1, errors='ignore')
        y_orig = original_df[target_col]
        X_synth = synthetic_df.drop([target_col, 'id'], axis=1, errors='ignore')
        y_synth = synthetic_df[target_col]
        
        # Ensure columns match
        common_cols = list(set(X_orig.columns) & set(X_synth.columns))
        X_orig = X_orig[common_cols]
        X_synth = X_synth[common_cols]
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_orig, y_orig)
        
        # Predict on synthetic
        y_pred = model.predict(X_synth)
        
        # Calculate error
        mse = mean_squared_error(y_synth, y_pred)
        print(f"Predictive MSE (train on original, predict on synthetic): {mse:.4f}")
        
        # Now train on synthetic, predict on original
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_synth, y_synth)
        
        # Predict on original
        y_pred = model.predict(X_orig)
        
        # Calculate error
        mse = mean_squared_error(y_orig, y_pred)
        print(f"Predictive MSE (train on synthetic, predict on original): {mse:.4f}")
    
    return comparison

def prepare_synthetic_data_for_validation(synthetic_df, format_type='binary'):
    """
    Prepare synthetic data for validation with submission files
    
    Args:
        synthetic_df: DataFrame with synthetic data
        format_type: How to format the target variable ('binary', 'continuous', 'probability')
    
    Returns:
        Prepared DataFrame ready for validation
    """
    if synthetic_df is None or synthetic_df.empty or 'rainfall' not in synthetic_df.columns:
        print("Cannot prepare invalid synthetic data")
        return synthetic_df
    
    # Make a copy to avoid modifying original
    df = synthetic_df.copy()
    
    if format_type == 'binary':
        # Convert rainfall to binary using median as threshold
        threshold = df['rainfall'].median()
        df['rainfall_original'] = df['rainfall'].copy()
        df['rainfall'] = (df['rainfall'] > threshold).astype(int)
        print(f"Converted rainfall to binary using threshold {threshold}")
    
    elif format_type == 'probability':
        # Scale rainfall to 0-1 range if not already
        if df['rainfall'].min() < 0 or df['rainfall'].max() > 1:
            df['rainfall_original'] = df['rainfall'].copy()
            df['rainfall'] = (df['rainfall'] - df['rainfall'].min()) / (df['rainfall'].max() - df['rainfall'].min())
            print("Scaled rainfall to 0-1 probability range")
    
    # Ensure we have an ID column for matching
    if 'id' not in df.columns:
        df['id'] = np.arange(len(df)) + 1
        print("Added ID column to synthetic data")
    
    return df

def generate_synthetic_data(method='bootstrap', n_samples=1000, format_for_validation=False):
    """Generate synthetic data using specified method"""
    print(f"Generating synthetic data using {method} method...")
    
    # Load original data
    train_df, _ = load_data()
    train_df, _ = preprocess_data(train_df, None)
    
    # Choose generation method
    if method == 'bootstrap':
        synthetic_df = bootstrap_sampling(train_df, n_samples)
    elif method == 'parametric':
        synthetic_df = parametric_sampling(train_df, n_samples)
    elif method == 'copula':
        synthetic_df = copula_based_sampling(train_df, n_samples)
    elif method == 'perturbation':
        synthetic_df = perturbation_based_sampling(train_df, n_samples)
    elif method == 'weather':
        synthetic_df = weather_pattern_simulation(train_df, n_samples)
    else:
        print(f"Unknown method: {method}. Using bootstrap sampling instead.")
        synthetic_df = bootstrap_sampling(train_df, n_samples)
    
    # Evaluate quality
    if not synthetic_df.empty:
        evaluation = evaluate_synthetic_data(train_df, synthetic_df)
        
        # Optionally format for validation
        if format_for_validation:
            synthetic_df = prepare_synthetic_data_for_validation(synthetic_df, format_type='binary')
        
        # Save synthetic data
        synthetic_df.to_csv(f'/home/USER/Projects/RAIN/data/synthetic_{method}.csv', index=False)
        print(f"Saved {len(synthetic_df)} synthetic samples to '/home/USER/Projects/RAIN/data/synthetic_{method}.csv'")
    else:
        print("No synthetic data generated.")
    
    return synthetic_df

def generate_cnn_training_data(base_samples=10000, augmentation_factor=3):
    """
    Generate a large synthetic dataset suitable for training CNNs
    
    Args:
        base_samples: Number of base samples to generate
        augmentation_factor: How much to augment the data
        
    Returns:
        DataFrame containing the synthetic data
    """
    print(f"Generating CNN training dataset with {base_samples} base samples...")
    
    # Generate base samples using ensemble approach (multiple methods)
    methods = ['bootstrap', 'parametric', 'weather', 'perturbation']
    samples_per_method = base_samples // len(methods)
    
    # Collect dataframes from each method
    all_dfs = []
    for method in methods:
        print(f"Generating {samples_per_method} samples using {method} method...")
        df = generate_synthetic_data(method=method, n_samples=samples_per_method)
        if not df.empty:
            all_dfs.append(df)
    
    # Combine all methods
    if not all_dfs:
        print("Failed to generate any synthetic data")
        return pd.DataFrame()
        
    base_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Generated {len(base_df)} base samples")
    
    # If augmentation is requested
    if augmentation_factor > 1:
        # Create augmented versions by adding noise to continuous features
        augmented_dfs = [base_df]
        
        for i in range(augmentation_factor - 1):
            print(f"Creating augmented dataset {i+1}/{augmentation_factor-1}...")
            aug_df = base_df.copy()
            
            # Add varying levels of noise to continuous features
            for col in aug_df.columns:
                if aug_df[col].dtype in ['float64', 'int64'] and col not in ['id', 'day']:
                    noise_scale = aug_df[col].std() * (0.05 + i * 0.02)  # Increasing noise
                    aug_df[col] = aug_df[col] + np.random.normal(0, noise_scale, len(aug_df))
            
            # Add slight variations to rainfall based on other features
            if 'rainfall' in aug_df.columns:
                # Add small random adjustments based on feature correlations
                noise = np.random.normal(0, aug_df['rainfall'].std() * 0.05, len(aug_df))
                aug_df['rainfall'] = aug_df['rainfall'] + noise
                # Ensure no negative values
                aug_df['rainfall'] = np.maximum(aug_df['rainfall'], 0)
                
            augmented_dfs.append(aug_df)
        
        # Combine all augmented versions
        final_df = pd.concat(augmented_dfs, ignore_index=True)
        print(f"Generated a total of {len(final_df)} samples with augmentation")
    else:
        final_df = base_df
    
    # Create new IDs for the combined dataset
    if 'id' in final_df.columns:
        final_df['id'] = np.arange(len(final_df)) + 1
    
    # Save the large dataset
    final_df.to_csv('/home/USER/Projects/RAIN/data/synthetic_cnn_training.csv', index=False)
    print(f"Saved {len(final_df)} synthetic samples to '/home/USER/Projects/RAIN/data/synthetic_cnn_training.csv'")
    
    return final_df

def prepare_data_for_cnn(df, sequence_length=30, target_col='rainfall'):
    """
    Prepare data in a format suitable for CNN training
    
    Args:
        df: DataFrame containing the data
        sequence_length: Length of sequences for CNN input
        target_col: Target column to predict
        
    Returns:
        X: Input sequences for CNN
        y: Target values
    """
    if 'day' not in df.columns:
        print("Error: Day column required for sequence preparation")
        return None, None
    
    # Sort by day to ensure proper sequences
    df = df.sort_values('day')
    
    # Get features and target
    features = df.drop(['id', target_col], axis=1, errors='ignore')
    target = df[target_col].values
    
    # Create sequences
    X = []
    y = []
    
    # For each valid sequence starting point
    for i in range(len(df) - sequence_length + 1):
        # Get sequence of features
        seq = features.iloc[i:i+sequence_length].values
        # Get corresponding target (using the last day in the sequence)
        target_val = target[i+sequence_length-1]
        
        X.append(seq)
        y.append(target_val)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created {len(X)} sequences of length {sequence_length}")
    print(f"Input shape: {X.shape}, Target shape: {y.shape}")
    
    return X, y

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic data for rainfall prediction')
    parser.add_argument('--method', type=str, default='bootstrap', 
                       choices=['bootstrap', 'parametric', 'copula', 'perturbation', 'weather', 'cnn'],
                       help='Method to generate synthetic data')
    parser.add_argument('--samples', type=int, default=1000, 
                       help='Number of samples to generate')
    
    args = parser.parse_args()
    
    if args.method == 'cnn':
        # Generate large dataset for CNN
        synthetic_df = generate_cnn_training_data(base_samples=args.samples, augmentation_factor=3)
        
        # Optionally, prepare sequences for CNN training
        if not synthetic_df.empty and 'day' in synthetic_df.columns:
            X, y = prepare_data_for_cnn(synthetic_df, sequence_length=30)
            
            # Save prepared data in numpy format for direct loading into CNN
            import os
            os.makedirs('/home/USER/Projects/RAIN/data/cnn_data', exist_ok=True)
            np.save('/home/USER/Projects/RAIN/data/cnn_data/X_synthetic.npy', X)
            np.save('/home/USER/Projects/RAIN/data/cnn_data/y_synthetic.npy', y)
            print("Saved CNN-ready data to '/home/USER/Projects/RAIN/data/cnn_data/'")
    else:
        # Original behavior for standard methods
        # Analyze original data distributions
        top_features, correlations = analyze_data_distributions()
        print("Top features correlated with rainfall:")
        for feature in top_features[:5]:
            print(f"  {feature}: {correlations[feature]:.4f}")
        
        print(f"\n{'-'*50}")
        print(f"Generating synthetic data using {args.method} method")
        print(f"{'-'*50}")
        synthetic_df = generate_synthetic_data(method=args.method, n_samples=args.samples)
