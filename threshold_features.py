import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import os

def create_threshold_features(train, test):
    """Create features optimized for distinguishing rainfall thresholds"""
    X = train.copy()
    X_test = test.copy()
    
    # Extract essential columns
    feature_cols = [col for col in X.columns if col not in ['id', 'rainfall']]
    
    # Basic features
    for col in feature_cols:
        # Create squared and sqrt versions for all features
        X[f'{col}_squared'] = X[col] ** 2
        X_test[f'{col}_squared'] = X_test[col] ** 2
        
        if (X[col] >= 0).all():
            X[f'{col}_sqrt'] = np.sqrt(np.maximum(0, X[col]))
            X_test[f'{col}_sqrt'] = np.sqrt(np.maximum(0, X_test[col]))
    
    # Create weather pattern indicators
    # 1. Humidity-Temperature relationship (high humidity + high temp = rain potential)
    humidity_col = next((col for col in feature_cols if 'humid' in col.lower()), None)
    temp_col = next((col for col in feature_cols if 'temp' in col.lower()), None)
    
    if humidity_col and temp_col:
        X['humid_temp'] = X[humidity_col] * X[temp_col]
        X_test['humid_temp'] = X_test[humidity_col] * X_test[temp_col]
        
        # Rain threshold indicator based on humidity
        X['high_humidity'] = (X[humidity_col] > X[humidity_col].mean()).astype(int)
        X_test['high_humidity'] = (X_test[humidity_col] > X[humidity_col].mean()).astype(int)
    
    # 2. Cloud cover indicators
    cloud_col = next((col for col in feature_cols if 'cloud' in col.lower()), None)
    if cloud_col:
        # Binary indicators for heavy cloud cover
        X['heavy_cloud'] = (X[cloud_col] > X[cloud_col].quantile(0.7)).astype(int)
        X_test['heavy_cloud'] = (X_test[cloud_col] > X[cloud_col].quantile(0.7)).astype(int)
        
        # Very light cloud cover
        X['light_cloud'] = (X[cloud_col] < X[cloud_col].quantile(0.3)).astype(int)
        X_test['light_cloud'] = (X_test[cloud_col] < X[cloud_col].quantile(0.3)).astype(int)
    
    # 3. Seasonal rain patterns
    if 'id' in X.columns:
        X['day_of_year'] = X['id'] % 365
        X_test['day_of_year'] = X_test['id'] % 365
        
        X['month'] = (X['day_of_year'] // 30) + 1
        X_test['month'] = (X_test['day_of_year'] // 30) + 1
        
        # Season indicators (assuming standard seasons)
        X['winter'] = ((X['month'] == 12) | (X['month'] <= 2)).astype(int)
        X['spring'] = ((X['month'] >= 3) & (X['month'] <= 5)).astype(int)
        X['summer'] = ((X['month'] >= 6) & (X['month'] <= 8)).astype(int)
        X['fall'] = ((X['month'] >= 9) & (X['month'] <= 11)).astype(int)
        
        X_test['winter'] = ((X_test['month'] == 12) | (X_test['month'] <= 2)).astype(int)
        X_test['spring'] = ((X_test['month'] >= 3) & (X_test['month'] <= 5)).astype(int)
        X_test['summer'] = ((X_test['month'] >= 6) & (X_test['month'] <= 8)).astype(int)
        X_test['fall'] = ((X_test['month'] >= 9) & (X_test['month'] <= 11)).astype(int)
        
        # Seasonal interactions with weather features
        for col in feature_cols[:3]:  # Top features
            for season in ['winter', 'spring', 'summer', 'fall']:
                X[f'{col}_{season}'] = X[col] * X[season]
                X_test[f'{col}_{season}'] = X_test[col] * X_test[season]
    
    # Final cleanup
    X_final = X.drop(['id', 'rainfall'], axis=1, errors='ignore')
    X_test_final = X_test.drop(['id'], axis=1, errors='ignore')
    
    # Handle any missing values
    X_final = X_final.fillna(X_final.mean())
    X_test_final = X_test_final.fillna(X_final.mean())
    
    return X_final, X_test_final

def create_threshold_submission():
    """Create submission using threshold-optimized features"""
    # Load data
    train = pd.read_csv('/home/USER/Projects/RAIN/data/train.csv')
    test = pd.read_csv('/home/USER/Projects/RAIN/data/test.csv')
    
    # Generate features
    X, X_test = create_threshold_features(train, test)
    y = train['rainfall']
    
    print(f"Created {X.shape[1]} threshold-optimized features")
    
    # Create classifier
    clf = RandomForestClassifier(
        n_estimators=1500,
        max_depth=12,
        min_samples_split=3,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    
    # Cross-validation with AUC scoring
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_proba = cross_val_predict(clf, X, y, cv=kf, method='predict_proba')[:,1]
    auc_score = roc_auc_score(y, y_pred_proba)
    print(f"Cross-validation AUC: {auc_score:.6f}")
    
    # Train final model
    print("Training final model...")
    clf.fit(X, y)
    
    # Generate predictions
    test_proba = clf.predict_proba(X_test)[:,1]
    
    # Create submission
    submission = pd.DataFrame({
        'id': test['id'],
        'rainfall': test_proba
    })
    
    output_path = f"data/candidates/threshold_rf_auc_{auc_score:.6f}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

if __name__ == "__main__":
    create_threshold_submission()