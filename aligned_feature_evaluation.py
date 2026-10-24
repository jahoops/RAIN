import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, KFold
import os

def binary_feature_evaluation(name, feature_func):
    """Evaluate features using binary classification and ROC AUC scoring"""
    print(f"Evaluating {name} feature set with binary classification approach")
    
    # Load data
    train = pd.read_csv('/home/USER/Projects/RAIN/data/train.csv')
    test = pd.read_csv('/home/USER/Projects/RAIN/data/test.csv')
    
    # Generate features
    X, X_test = feature_func(train, test)
    y = train['rainfall']
    
    # Consider the task as binary classification
    # Use RandomForestClassifier instead of Regressor
    model = RandomForestClassifier(
        n_estimators=1000, 
        max_depth=12,
        min_samples_split=3,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    
    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_proba = cross_val_predict(model, X, y, cv=kf, method='predict_proba')[:,1]
    
    # Calculate AUC
    auc_score = roc_auc_score(y, y_pred_proba)
    print(f"Cross-validation AUC: {auc_score:.6f}")
    
    # Train final model and generate predictions
    model.fit(X, y)
    test_pred_proba = model.predict_proba(X_test)[:,1]
    
    # Create submission
    submission = pd.DataFrame({
        'id': test['id'],
        'rainfall': test_pred_proba
    })
    
    os.makedirs('data/binary_submissions', exist_ok=True)
    output_path = f"data/binary_submissions/{name}_auc_{auc_score:.6f}.csv"
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    return auc_score