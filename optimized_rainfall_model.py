import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os

def create_optimized_features(train, test):
    """Create an optimized feature set based on importance analysis"""
    X = train.copy()
    X_test = test.copy()
    
    # Extract essential columns (removing ID & target for processing)
    feature_cols = [col for col in X.columns if col not in ['id', 'rainfall']]
    
    # 1. Create non-linear transformations of top features
    key_features = ['cloud', 'sunshine', 'humidity', 'windspeed', 'pressure', 'dewpoint']
    
    for col in feature_cols:
        for key in key_features:
            if key in col.lower():
                # Create squared and square root versions
                X[f'{col}_squared'] = X[col] ** 2
                X_test[f'{col}_squared'] = X_test[col] ** 2
                
                if (X[col] >= 0).all():
                    X[f'{col}_sqrt'] = np.sqrt(np.maximum(0, X[col]))
                    X_test[f'{col}_sqrt'] = np.sqrt(np.maximum(0, X_test[col]))
    
    # 2. Add temporal features (best performing category)
    if 'id' in X.columns:
        # Day of year (cyclic)
        X['day_of_year'] = X['id'] % 365
        X_test['day_of_year'] = X_test['id'] % 365
        
        # Sine and cosine transform for cyclical patterns
        X['day_sin'] = np.sin(2 * np.pi * X['day_of_year']/365)
        X['day_cos'] = np.cos(2 * np.pi * X['day_of_year']/365)
        X_test['day_sin'] = np.sin(2 * np.pi * X_test['day_of_year']/365)
        X_test['day_cos'] = np.cos(2 * np.pi * X_test['day_of_year']/365)
        
        # Season indicators (since season was in the importance list)
        X['month'] = ((X['day_of_year'] // 30) + 1).clip(1, 12) 
        X_test['month'] = ((X_test['day_of_year'] // 30) + 1).clip(1, 12)
        X['season'] = ((X['month'] - 1) // 3) + 1
        X_test['season'] = ((X_test['month'] - 1) // 3) + 1
    
    # 3. Create targeted interactions between top features ONLY
    # Based on importance rankings, certain interactions matter
    top_features = ['cloud', 'sunshine', 'humidity']
    secondary_features = ['pressure', 'dewpoint', 'temperature', 'maxtemp', 'mintemp']
    
    for feat1 in top_features:
        for feat2 in secondary_features:
            col1 = next((c for c in feature_cols if feat1 in c.lower()), None)
            col2 = next((c for c in feature_cols if feat2 in c.lower()), None)
            
            if col1 and col2:
                X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
                X_test[f'{col1}_x_{col2}'] = X_test[col1] * X_test[col2]
    
    # Final cleanup
    X_final = X.drop(['id', 'rainfall'], axis=1, errors='ignore')
    X_test_final = X_test.drop(['id'], axis=1, errors='ignore')
    
    # Handle any missing values
    X_final = X_final.fillna(X_final.mean())
    X_test_final = X_test_final.fillna(X_final.mean())
    
    return X_final, X_test_final

def create_optimized_submission():
    """Create a submission using our optimized feature strategy with CV score in filename"""
    from sklearn.model_selection import KFold, cross_val_score
    
    # Load data
    train = pd.read_csv('/home/USER/Projects/RAIN/data/train.csv')
    test = pd.read_csv('/home/USER/Projects/RAIN/data/test.csv')
    
    # Generate optimized features
    X, X_test = create_optimized_features(train, test)
    y = train['rainfall']
    
    print(f"Created {X.shape[1]} optimized features")
    
    # Define model
    rf = RandomForestRegressor(
        n_estimators=1200,
        max_depth=12,
        min_samples_split=3,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    
    # Calculate cross-validation score
    print("Calculating cross-validation score...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=kf, scoring='neg_root_mean_squared_error')
    rmse_score = -cv_scores.mean()
    print(f"Cross-validation RMSE: {rmse_score:.6f}")
    
    # Train final model
    print("Training final model...")
    rf.fit(X, y)
    
    # Generate predictions
    predictions = rf.predict(X_test)
    
    # Create submission with score in filename
    score_str = f"score_{rmse_score:.6f}"
    output_path = f"data/candidates/rf_optimized_features_{score_str}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    submission = pd.DataFrame({
        'id': test['id'],
        'rainfall': predictions
    })
    
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    # Also save feature importances for reference
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importances.to_csv(f"data/candidates/rf_optimized_features_{score_str}_importances.csv", index=False)
    print(f"Feature importances saved")

if __name__ == "__main__":
    create_optimized_submission()