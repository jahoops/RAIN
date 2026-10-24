import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import os
from optimized_binary_model import create_binary_optimized_features

def create_gradient_boosting_submission():
    """Create a submission using gradient boosting classifier"""
    # Load data
    train = pd.read_csv('/home/USER/Projects/RAIN/data/train.csv')
    test = pd.read_csv('/home/USER/Projects/RAIN/data/test.csv')
    
    # Generate features using the same function as before
    X, X_test = create_binary_optimized_features(train, test)
    y = train['rainfall']
    
    # Create gradient boosting classifier
    gbc = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    )
    
    # Cross-validation with AUC scoring
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_proba = cross_val_predict(gbc, X, y, cv=kf, method='predict_proba')[:,1]
    auc_score = roc_auc_score(y, y_pred_proba)
    print(f"Cross-validation AUC: {auc_score:.6f}")
    
    # Train final model
    print("Training final model...")
    gbc.fit(X, y)
    
    # Generate prediction probabilities
    test_proba = gbc.predict_proba(X_test)[:,1]
    
    # Create submission
    submission = pd.DataFrame({
        'id': test['id'],
        'rainfall': test_proba
    })
    
    output_path = f"data/candidates/gb_auc_{auc_score:.6f}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

if __name__ == "__main__":
    create_gradient_boosting_submission()