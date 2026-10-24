# Add these to the create_binary_optimized_features function

from optimized_binary_model import create_binary_optimized_features
import numpy as np

def enhanced_binary_features(train, test):
    """Add additional meteorological transformations"""
    X, X_test = create_binary_optimized_features(train, test)
    
    # Extract essential columns for weather relationships
    feature_cols = [col for col in train.columns if col not in ['id', 'rainfall']]
    
    # Find key weather columns
    temp_col = next((col for col in feature_cols if 'temp' in col.lower()), None)
    humid_col = next((col for col in feature_cols if 'humid' in col.lower()), None)
    cloud_col = next((col for col in feature_cols if 'cloud' in col.lower()), None)
    
    # Weather-specific transformations
    if temp_col and humid_col:
        # Vapor pressure deficit proxy (temp & humidity relationship)
        X['vpd'] = X[temp_col] * (100 - X[humid_col]) / 100
        X_test['vpd'] = X_test[temp_col] * (100 - X_test[humid_col]) / 100
        
        # Heat index approximation
        X['heat_index'] = X[temp_col] + 0.05 * X[humid_col]
        X_test['heat_index'] = X_test[temp_col] + 0.05 * X_test[humid_col]
        
    if cloud_col and temp_col:
        # Cloud-temperature interaction (affects precipitation likelihood)
        X['cloud_temp'] = X[cloud_col] * X[temp_col]
        X_test['cloud_temp'] = X_test[cloud_col] * X_test[temp_col]
    
    # Try cubic transformations for top features
    top_features = ['cloud_squared', 'sunshine_sqrt', 'sunshine', 'dewpoint_squared']
    for col in top_features:
        if col in X.columns:
            X[f'{col}_cubed'] = X[col] ** 3
            X_test[f'{col}_cubed'] = X_test[col] ** 3
    
    # Log transforms with offset to avoid log(0)
    for col in ['cloud', 'humidity']:
        if col in X.columns:
            X[f'{col}_log'] = np.log1p(X[col])
            X_test[f'{col}_log'] = np.log1p(X_test[col])
    
    return X, X_test