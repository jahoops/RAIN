import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def basic_features(train, test):
    """Basic feature set with minimal processing"""
    X = train.drop(['id', 'rainfall'], axis=1)
    X_test = test.drop(['id'], axis=1)
    
    # Align columns
    for col in set(X.columns) - set(X_test.columns):
        X_test[col] = 0
    X_test = X_test[X.columns]
    
    new_features = []  # No new features
    return X, X_test, new_features

def temporal_features(train, test):
    """Add temporal/seasonal features"""
    X = train.drop(['rainfall'], axis=1).copy()
    X_test = test.copy()
    
    new_features = []
    
    # Create month from ID (assuming IDs relate to days in sequence)
    if 'id' in X.columns:
        X['month'] = ((X['id'] % 365) // 30) + 1
        X_test['month'] = ((X_test['id'] % 365) // 30) + 1
        new_features.append('month')
        
        # Season (1-4)
        X['season'] = ((X['month'] - 1) // 3) + 1
        X_test['season'] = ((X_test['month'] - 1) // 3) + 1
        new_features.append('season')
        
        # Day of year
        X['day_of_year'] = X['id'] % 365
        X_test['day_of_year'] = X_test['id'] % 365
        new_features.append('day_of_year')
        
        # Cyclic encoding of day of year
        X['day_sin'] = np.sin(2 * np.pi * X['day_of_year']/365)
        X_test['day_sin'] = np.sin(2 * np.pi * X_test['day_of_year']/365)
        new_features.append('day_sin')
        
        X['day_cos'] = np.cos(2 * np.pi * X['day_of_year']/365)
        X_test['day_cos'] = np.cos(2 * np.pi * X_test['day_of_year']/365)
        new_features.append('day_cos')
        
    # Drop ID column for modeling
    if 'id' in X.columns:
        X = X.drop(['id'], axis=1)
    if 'id' in X_test.columns:
        X_test = X_test.drop(['id'], axis=1)
    
    # Align columns
    for col in set(X.columns) - set(X_test.columns):
        X_test[col] = 0
    X_test = X_test[X.columns]
    
    return X, X_test, new_features

def statistical_transformations(train, test):
    """Apply statistical transformations to features"""
    X = train.drop(['rainfall'], axis=1).copy()
    X_test = test.copy()
    
    # Get numerical columns excluding ID
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    if 'id' in num_cols:
        num_cols.remove('id')
        
    new_features = []
    
    # Log transform for skewed features
    for col in num_cols:
        # Check if data is positive
        if (X[col] > 0).all():
            X[f'{col}_log'] = np.log1p(X[col])
            X_test[f'{col}_log'] = np.log1p(X_test[col])
            new_features.append(f'{col}_log')
    
    # Square and square root
    for col in num_cols:
        X[f'{col}_squared'] = X[col] ** 2
        X_test[f'{col}_squared'] = X_test[col] ** 2
        new_features.append(f'{col}_squared')
        
        # Square root for non-negative values
        if (X[col] >= 0).all():
            X[f'{col}_sqrt'] = np.sqrt(X[col])
            X_test[f'{col}_sqrt'] = np.sqrt(X_test[col])
            new_features.append(f'{col}_sqrt')
    
    # Z-score normalization
    for col in num_cols:
        mean = X[col].mean()
        std = X[col].std()
        if std > 0:
            X[f'{col}_z'] = (X[col] - mean) / std
            X_test[f'{col}_z'] = (X_test[col] - mean) / std
            new_features.append(f'{col}_z')
    
    # Min-max scaling
    for col in num_cols:
        min_val = X[col].min()
        max_val = X[col].max()
        range_val = max_val - min_val
        if range_val > 0:
            X[f'{col}_scaled'] = (X[col] - min_val) / range_val
            X_test[f'{col}_scaled'] = (X_test[col] - min_val) / range_val
            new_features.append(f'{col}_scaled')
    
    # Drop ID column for modeling
    if 'id' in X.columns:
        X = X.drop(['id'], axis=1)
    if 'id' in X_test.columns:
        X_test = X_test.drop(['id'], axis=1)
    
    # Align columns
    for col in set(X.columns) - set(X_test.columns):
        X_test[col] = 0
    X_test = X_test[X.columns]
    
    return X, X_test, new_features

def polynomial_features(train, test):
    """Create polynomial and interaction features"""
    X = train.drop(['rainfall'], axis=1).copy()
    X_test = test.copy()
    
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    if 'id' in num_cols:
        num_cols.remove('id')
    
    new_features = []
    
    # Select top features by correlation with target
    target = train['rainfall']
    correlations = {}
    for col in num_cols:
        correlations[col] = np.abs(np.corrcoef(X[col], target)[0, 1])
    
    # Get top 5 features by correlation
    top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:5]
    top_cols = [col for col, corr in top_features]
    
    # Create interactions between top features
    for i in range(len(top_cols)):
        for j in range(i+1, len(top_cols)):
            col1 = top_cols[i]
            col2 = top_cols[j]
            interaction_name = f'{col1}_x_{col2}'
            X[interaction_name] = X[col1] * X[col2]
            X_test[interaction_name] = X_test[col1] * X_test[col2]
            new_features.append(interaction_name)
            
            # Ratio features (if denominator is not zero)
            if (X[col2] != 0).all():
                ratio_name = f'{col1}_div_{col2}'
                X[ratio_name] = X[col1] / (X[col2] + 1e-8)
                X_test[ratio_name] = X_test[col1] / (X_test[col2] + 1e-8)
                new_features.append(ratio_name)
            
            if (X[col1] != 0).all():
                ratio_name = f'{col2}_div_{col1}'
                X[ratio_name] = X[col2] / (X[col1] + 1e-8)
                X_test[ratio_name] = X_test[col2] / (X_test[col1] + 1e-8)
                new_features.append(ratio_name)
    
    # Drop ID column for modeling
    if 'id' in X.columns:
        X = X.drop(['id'], axis=1)
    if 'id' in X_test.columns:
        X_test = X_test.drop(['id'], axis=1)
    
    # Align columns
    for col in set(X.columns) - set(X_test.columns):
        X_test[col] = 0
    X_test = X_test[X.columns]
    
    return X, X_test, new_features

def dimensionality_reduction(train, test):
    """Apply dimensionality reduction techniques"""
    X = train.drop(['rainfall'], axis=1).copy()
    X_test = test.copy()
    
    # Store IDs
    train_ids = X['id'] if 'id' in X.columns else None
    test_ids = X_test['id'] if 'id' in X_test.columns else None
    
    # Drop ID column for dimensionality reduction
    if 'id' in X.columns:
        X = X.drop(['id'], axis=1)
    if 'id' in X_test.columns:
        X_test = X_test.drop(['id'], axis=1)
    
    # Handle missing values BEFORE scaling and PCA
    X = X.fillna(X.mean())
    X_test = X_test.fillna(X.mean())
    
    # Scale data for PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    new_features = []
    
    # Apply PCA with error handling
    for n_components in [5, 10, 15]:
        if n_components < min(X.shape[1], X.shape[0]):
            try:
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X_scaled)
                X_test_pca = pca.transform(X_test_scaled)
                
                # Add PCA components as features
                for i in range(n_components):
                    col_name = f'pca_{n_components}_{i}'
                    X[col_name] = X_pca[:, i]
                    X_test[col_name] = X_test_pca[:, i]
                    new_features.append(col_name)
            except Exception as e:
                print(f"PCA with {n_components} components failed: {e}")
                continue
    
    # Add back IDs
    if train_ids is not None:
        X['id'] = train_ids
    if test_ids is not None:
        X_test['id'] = test_ids
    
    # Drop ID column for modeling
    if 'id' in X.columns:
        X = X.drop(['id'], axis=1)
    if 'id' in X_test.columns:
        X_test = X_test.drop(['id'], axis=1)
    
    # Align columns
    for col in set(X.columns) - set(X_test.columns):
        X_test[col] = 0
    X_test = X_test[X.columns]
    
    return X, X_test, new_features

def domain_specific_features(train, test):
    """Create domain-specific features for weather/rainfall prediction"""
    X = train.drop(['rainfall'], axis=1).copy()
    X_test = test.copy()
    
    new_features = []
    
    # Assuming we have temperature and humidity columns
    weather_cols = [col for col in X.columns if any(term in col.lower() for term in 
                                                  ['temp', 'humidity', 'pressure', 'wind', 'cloud'])]
    
    # If no weather columns found, try to use all numeric columns
    if not weather_cols:
        print("No weather columns found. Using all numeric columns.")
        weather_cols = X.select_dtypes(include=np.number).columns.tolist()
        if 'id' in weather_cols:
            weather_cols.remove('id')
    
    # Create composite features
    if len(weather_cols) >= 2:
        # Create interactions between top features
        top_cols = weather_cols[:min(5, len(weather_cols))]
        for i in range(len(top_cols)):
            for j in range(i+1, len(top_cols)):
                col1 = top_cols[i]
                col2 = top_cols[j]
                
                interaction_name = f'{col1}_x_{col2}'
                X[interaction_name] = X[col1] * X[col2]
                X_test[interaction_name] = X_test[col1] * X_test[col2]
                new_features.append(interaction_name)
    
    # Create historical features if ID represents time sequence
    if 'id' in X.columns:
        # Sort by ID to ensure temporal order
        X = X.sort_values('id').reset_index(drop=True)
        
        # Create lag features for selected columns
        for col in weather_cols[:min(3, len(weather_cols))]:  # Limit to top 3 columns to avoid dimensionality issues
            # Create lag features
            X[f'{col}_lag1'] = X[col].shift(1)
            new_features.append(f'{col}_lag1')
            
            # Create moving averages
            X[f'{col}_ma3'] = X[col].rolling(3, min_periods=1).mean()
            new_features.append(f'{col}_ma3')
    
    # Handle test set lag features safely
    for feature in new_features:
        if feature not in X_test.columns:
            X_test[feature] = np.nan
    
    # Drop ID column for modeling
    if 'id' in X.columns:
        X = X.drop(['id'], axis=1)
    if 'id' in X_test.columns:
        X_test = X_test.drop(['id'], axis=1)
    
    # Handle NaN values - CRITICAL
    X = X.fillna(X.mean())
    X_test = X_test.fillna(X_test.mean())
    
    # Align columns
    for col in set(X.columns) - set(X_test.columns):
        X_test[col] = 0
    X_test = X_test[X.columns]
    
    return X, X_test, new_features

def combined_features(train, test):
    """Combine multiple feature sets for maximum performance"""
    # Start with basic features
    X, X_test, basic_new = basic_features(train, test)
    
    # Add temporal features
    temp_train = train.copy()
    temp_test = test.copy()
    _, _, temp_new = temporal_features(temp_train, temp_test)
    for feature in temp_new:
        X[feature] = temp_train[feature]
        X_test[feature] = temp_test[feature]
    
    # Add statistical transformations
    stat_train = train.copy()
    stat_test = test.copy()
    _, _, stat_new = statistical_transformations(stat_train, stat_test)
    
    # Only add the most important statistical features to avoid dimensionality issues
    important_stat = stat_new[:5]  # Take top 5 statistical features
    for feature in important_stat:
        X[feature] = stat_train[feature]
        X_test[feature] = stat_test[feature]
    
    # Add domain specific features
    domain_train = train.copy()
    domain_test = test.copy()
    _, _, domain_new = domain_specific_features(domain_train, domain_test)
    for feature in domain_new:
        if feature in domain_train.columns:
            X[feature] = domain_train[feature]
            X_test[feature] = domain_test[feature]
    
    # Combine all new features
    new_features = basic_new + temp_new + important_stat + domain_new
    
    # Drop ID column for modeling
    if 'id' in X.columns:
        X = X.drop(['id'], axis=1)
    if 'id' in X_test.columns:
        X_test = X_test.drop(['id'], axis=1)
    
    # Handle any missing values
    X = X.fillna(X.mean())
    X_test = X_test.fillna(X_test.mean())
    
    # Align columns
    for col in set(X.columns) - set(X_test.columns):
        X_test[col] = 0
    X_test = X_test[X.columns]
    
    return X, X_test, new_features