import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os

def create_rainfall_specific_submission(name, feature_func):
    """Create a submission using specialized rainfall features"""
    # Load data
    train = pd.read_csv('/home/USER/Projects/RAIN/data/train.csv')
    test = pd.read_csv('/home/USER/Projects/RAIN/data/test.csv')
    
    # Apply feature engineering
    print(f"Applying {name} feature engineering...")
    X, X_test = feature_func(train, test)
    
    # Prepare target
    y = train['rainfall']
    
    # Train optimized Random Forest
    print("Training Random Forest model...")
    rf = RandomForestRegressor(
        n_estimators=1000,  # More trees for better performance
        max_depth=12,       # Slightly deeper trees
        min_samples_split=3,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X, y)
    
    # Generate predictions
    predictions = rf.predict(X_test)
    
    # Save submission
    submission = pd.DataFrame({
        'id': test['id'],
        'rainfall': predictions
    })
    
    os.makedirs('data/candidates', exist_ok=True)
    submission_path = f'data/candidates/rf_{name}_submission.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}")
    
    return submission_path

# 1. METEOROLOGICAL TRANSFORMATIONS
def meteorological_features(train, test):
    """Create specific features for meteorological prediction"""
    # Copy data
    X = train.copy()
    X_test = test.copy()
    
    # Extract feature columns and target
    feature_cols = [col for col in X.columns if col not in ['id', 'rainfall']]
    
    # Create cyclic day-of-year features
    if 'id' in X.columns:
        X['day_of_year'] = X['id'] % 365
        X_test['day_of_year'] = X_test['id'] % 365
        
        # Sine and cosine transform for cyclical patterns
        X['day_sin'] = np.sin(2 * np.pi * X['day_of_year']/365)
        X['day_cos'] = np.cos(2 * np.pi * X['day_of_year']/365)
        X_test['day_sin'] = np.sin(2 * np.pi * X_test['day_of_year']/365)
        X_test['day_cos'] = np.cos(2 * np.pi * X_test['day_of_year']/365)
        
        # Create month feature (1-12)
        X['month'] = ((X['day_of_year'] // 30) + 1).clip(1, 12)
        X_test['month'] = ((X_test['day_of_year'] // 30) + 1).clip(1, 12)
        
        # Season indicators
        X['season'] = ((X['month'] - 1) // 3) + 1  # 1-4 for seasons
        X_test['season'] = ((X_test['month'] - 1) // 3) + 1
        
        # Add season-specific indicators
        for s in range(1, 5):
            X[f'is_season_{s}'] = (X['season'] == s).astype(int)
            X_test[f'is_season_{s}'] = (X_test['season'] == s).astype(int)
    
    # Feature interactions focusing on weather patterns
    for i, col1 in enumerate(feature_cols[:10]):  # Limit to top 10 features
        # Add squared terms (important for meteorology)
        X[f'{col1}_squared'] = X[col1] ** 2
        X_test[f'{col1}_squared'] = X_test[col1] ** 2
        
        # Log transform for skewed distributions
        if (X[col1] > 0).all():
            X[f'{col1}_log'] = np.log1p(X[col1])
            X_test[f'{col1}_log'] = np.log1p(X_test[col1])
            
        # Interactions with season
        if 'season' in X.columns:
            for s in range(1, 5):
                X[f'{col1}_season{s}'] = X[col1] * X[f'is_season_{s}']
                X_test[f'{col1}_season{s}'] = X_test[col1] * X_test[f'is_season_{s}']
    
    # Prepare final feature matrices
    X_final = X.drop(['id', 'rainfall'], axis=1, errors='ignore')
    X_test_final = X_test.drop(['id'], axis=1, errors='ignore')
    
    # Align columns
    for col in set(X_final.columns) - set(X_test_final.columns):
        X_test_final[col] = 0
    for col in set(X_test_final.columns) - set(X_final.columns):
        X_final[col] = 0
    X_test_final = X_test_final[X_final.columns]
    
    # Handle any missing values
    X_final = X_final.fillna(X_final.mean())
    X_test_final = X_test_final.fillna(X_test_final.mean())
    
    return X_final, X_test_final

# 2. CUSTOM RAINFALL FEATURES
def rainfall_pattern_features(train, test):
    """Create features specifically for rainfall prediction"""
    X = train.copy()
    X_test = test.copy()
    
    # Basic feature columns
    feature_cols = [col for col in X.columns if col not in ['id', 'rainfall']]
    
    # Create lag features if ID is sequential
    if 'id' in X.columns:
        X = X.sort_values('id').reset_index(drop=True)
        X_test = X_test.sort_values('id').reset_index(drop=True)
        
        # Create month and season
        X['month'] = ((X['id'] % 365) // 30) + 1
        X_test['month'] = ((X_test['id'] % 365) // 30) + 1
        
        # Dry/wet season binary indicator
        dry_months = [1, 2, 11, 12]  # Example dry season months
        X['dry_season'] = X['month'].isin(dry_months).astype(int)
        X_test['dry_season'] = X_test['month'].isin(dry_months).astype(int)
        
        # Create polynomial features for key columns
        for col in feature_cols[:5]:  # Top 5 features
            X[f'{col}_squared'] = X[col] ** 2
            X_test[f'{col}_squared'] = X_test[col] ** 2
            X[f'{col}_cubed'] = X[col] ** 3
            X_test[f'{col}_cubed'] = X_test[col] ** 3
        
        # Feature ratios (important for meteorological predictions)
        if len(feature_cols) >= 2:
            for i in range(min(5, len(feature_cols))):
                for j in range(i+1, min(5, len(feature_cols))):
                    col1 = feature_cols[i]
                    col2 = feature_cols[j]
                    
                    # Create ratio features (with safeguards)
                    X[f'{col1}_div_{col2}'] = X[col1] / (X[col2] + 1e-8)
                    X_test[f'{col1}_div_{col2}'] = X_test[col1] / (X_test[col2] + 1e-8)
                    
                    # Sum and difference
                    X[f'{col1}_plus_{col2}'] = X[col1] + X[col2]
                    X_test[f'{col1}_plus_{col2}'] = X_test[col1] + X_test[col2]
    
    # Prepare final feature matrices
    X_final = X.drop(['id', 'rainfall'], axis=1, errors='ignore')
    X_test_final = X_test.drop(['id'], axis=1, errors='ignore')
    
    # Align columns
    for col in set(X_final.columns) - set(X_test_final.columns):
        X_test_final[col] = 0
    for col in set(X_test_final.columns) - set(X_final.columns):
        X_final[col] = 0
    X_test_final = X_test_final[X_final.columns]
    
    # Handle any missing values
    X_final = X_final.fillna(X_final.mean())
    X_test_final = X_test_final.fillna(X_test_final.mean())
    
    return X_final, X_test_final

# 3. CONDENSATION PHYSICS FEATURES
def condensation_physics_features(train, test):
    """Create features based on physics of rainfall formation"""
    X = train.copy()
    X_test = test.copy()
    
    feature_cols = [col for col in X.columns if col not in ['id', 'rainfall']]
    
    # Create temporal features
    if 'id' in X.columns:
        X['day_of_year'] = X['id'] % 365
        X_test['day_of_year'] = X_test['id'] % 365
        
        # Create cyclic encoding of day
        X['day_sin'] = np.sin(2 * np.pi * X['day_of_year']/365)
        X['day_cos'] = np.cos(2 * np.pi * X['day_of_year']/365)
        X_test['day_sin'] = np.sin(2 * np.pi * X_test['day_of_year']/365)
        X_test['day_cos'] = np.cos(2 * np.pi * X_test['day_of_year']/365)
    
    # Physics-inspired features
    # If we have temperature and humidity-like features, create dew point proxies
    temp_cols = [col for col in feature_cols if 'temp' in col.lower()]
    humid_cols = [col for col in feature_cols if 'hum' in col.lower()]
    
    if temp_cols and humid_cols:
        temp_col = temp_cols[0]
        humid_col = humid_cols[0]
        
        # Create dew point proxy
        X['dew_point_proxy'] = X[temp_col] * X[humid_col] / 100
        X_test['dew_point_proxy'] = X_test[temp_col] * X_test[humid_col] / 100
        
        # Create "comfort index" (heat index proxy)
        X['heat_index'] = X[temp_col] + 0.05 * X[humid_col]
        X_test['heat_index'] = X_test[temp_col] + 0.05 * X_test[humid_col]
        
        # Create lifted condensation level proxy
        X['lcl_proxy'] = 125 * (X[temp_col] - X['dew_point_proxy'])
        X_test['lcl_proxy'] = 125 * (X_test[temp_col] - X_test['dew_point_proxy'])
    
    # Create interaction features between all pairs of features
    for i, col1 in enumerate(feature_cols[:5]):
        for j in range(i+1, min(len(feature_cols), 6)):
            col2 = feature_cols[j]
            
            # Multiplication interaction
            X[f'{col1}_{col2}_interact'] = X[col1] * X[col2]
            X_test[f'{col1}_{col2}_interact'] = X_test[col1] * X_test[col2]
            
            # Custom physics-based transforms
            X[f'{col1}_{col2}_sqrt'] = np.sqrt(np.abs(X[col1] * X[col2]) + 1e-8)
            X_test[f'{col1}_{col2}_sqrt'] = np.sqrt(np.abs(X_test[col1] * X_test[col2]) + 1e-8)
            
            # Exponential decay physics (e.g., similar to vapor pressure)
            X[f'{col1}_{col2}_exp'] = np.exp(-np.abs(X[col1] - X[col2]))
            X_test[f'{col1}_{col2}_exp'] = np.exp(-np.abs(X_test[col1] - X_test[col2]))
    
    # Prepare final feature matrices
    X_final = X.drop(['id', 'rainfall'], axis=1, errors='ignore')
    X_test_final = X_test.drop(['id'], axis=1, errors='ignore')
    
    # Align columns
    for col in set(X_final.columns) - set(X_test_final.columns):
        X_test_final[col] = 0
    for col in set(X_test_final.columns) - set(X_final.columns):
        X_final[col] = 0
    X_test_final = X_test_final[X_final.columns]
    
    # Handle any missing values
    X_final = X_final.fillna(X_final.mean())
    X_test_final = X_test_final.fillna(X_test_final.mean())
    
    return X_final, X_test_final

# 4. COMBINED OPTIMIZED FEATURES
def optimized_combined_features(train, test):
    """Create optimized combined features from multiple approaches"""
    # Start with meteorological features
    X1, X1_test = meteorological_features(train, test)
    
    # Add rainfall pattern features
    X2, X2_test = rainfall_pattern_features(train, test)
    
    # Add condensation physics features
    X3, X3_test = condensation_physics_features(train, test)
    
    # Combine all features
    all_cols = set(X1.columns) | set(X2.columns) | set(X3.columns)
    
    # Create combined dataframes
    X = pd.DataFrame(index=X1.index)
    X_test = pd.DataFrame(index=X1_test.index)
    
    # Add columns from each source
    for col in all_cols:
        if col in X1.columns:
            X[col] = X1[col]
            X_test[col] = X1_test[col]
        elif col in X2.columns:
            X[col] = X2[col]
            X_test[col] = X2_test[col]
        elif col in X3.columns:
            X[col] = X3[col]
            X_test[col] = X3_test[col]
    
    # Handle any missing values
    X = X.fillna(X.mean())
    X_test = X_test.fillna(X_test.mean())
    
    return X, X_test

# 5. BORUTA SELECTED FEATURES WITH OPTIMIZATION
def boruta_optimized_features(train, test):
    """Use Boruta algorithm to select only the most important features"""
    from sklearn.ensemble import RandomForestRegressor
    from boruta import BorutaPy
    
    # Starting with optimized features
    X, X_test = optimized_combined_features(train, test)
    
    # Extract target
    y = train['rainfall']
    
    # Run Boruta feature selection with more permutations
    rf = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
    boruta = BorutaPy(rf, n_estimators=150, verbose=0, random_state=42)
    
    try:
        # Try to run Boruta
        boruta.fit(X.values, y.values)
        
        # Get selected features
        selected_features = X.columns[boruta.support_].tolist()
        print(f"Selected {len(selected_features)} features via Boruta")
        
        # Create new dataframes with only selected features
        X_selected = X[selected_features]
        X_test_selected = X_test[selected_features]
        
    except Exception as e:
        print(f"Boruta failed: {e}. Using all features instead.")
        X_selected = X
        X_test_selected = X_test
    
    return X_selected, X_test_selected

if __name__ == "__main__":
    # Create submission using meteorological features
    create_rainfall_specific_submission("meteorological", meteorological_features)
    
    # Create submission using rainfall pattern features
    create_rainfall_specific_submission("rainfall_patterns", rainfall_pattern_features)
    
    # Create submission using physics-based features
    create_rainfall_specific_submission("condensation_physics", condensation_physics_features)
    
    # Create submission using combined optimized features
    create_rainfall_specific_submission("combined_optimized", optimized_combined_features)
    
    # Create submission using Boruta-selected features
    try:
        create_rainfall_specific_submission("boruta_optimized", boruta_optimized_features)
    except ImportError:
        print("Boruta not available, skipping Boruta feature selection")