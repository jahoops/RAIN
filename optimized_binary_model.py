import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

def create_binary_optimized_features(train, test):
    """Create features optimized for binary classification of rainfall"""
    X = train.copy()
    X_test = test.copy()
    
    # Extract essential columns
    feature_cols = [col for col in X.columns if col not in ['id', 'rainfall']]
    
    # 1. Create non-linear transformations of top features from importance analysis
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
    
    # 3. Feature ratios and interactions
    for i, col1 in enumerate(feature_cols[:3]):  # Limit to top features
        for j in range(i+1, min(len(feature_cols), 4)):
            col2 = feature_cols[j]
            
            # Create interaction
            X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
            X_test[f'{col1}_x_{col2}'] = X_test[col1] * X_test[col2]
            
            # Create ratio with safeguard against division by zero
            X[f'{col1}_div_{col2}'] = X[col1] / (X[col2] + 1e-8)
            X_test[f'{col1}_div_{col2}'] = X_test[col1] / (X_test[col2] + 1e-8)
    
    # Final cleanup
    X_final = X.drop(['id', 'rainfall'], axis=1, errors='ignore')
    X_test_final = X_test.drop(['id'], axis=1, errors='ignore')
    
    # Handle any missing values
    X_final = X_final.fillna(X_final.mean())
    X_test_final = X_test_final.fillna(X_final.mean())
    
    return X_final, X_test_final

def create_binary_submission():
    """Create a submission using binary classification approach"""
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    
    # Load data
    train = pd.read_csv('/home/USER/Projects/RAIN/data/train.csv')
    test = pd.read_csv('/home/USER/Projects/RAIN/data/test.csv')
    
    # Generate features
    X, X_test = create_binary_optimized_features(train, test)
    y = train['rainfall']
    
    print(f"Created {X.shape[1]} optimized features for binary classification")
    
    # Create and train classifier
    clf = RandomForestClassifier(
        n_estimators=1200,
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
    
    # Generate prediction probabilities (not class predictions)
    test_proba = clf.predict_proba(X_test)[:,1]
    
    # Create submission
    submission = pd.DataFrame({
        'id': test['id'],
        'rainfall': test_proba  # Use probabilities, not class predictions
    })
    
    output_path = f"data/candidates/binary_rf_auc_{auc_score:.6f}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

if __name__ == "__main__":
    create_binary_submission()