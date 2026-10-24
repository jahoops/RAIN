# Just the core elements that worked best

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel
import os
from optimized_binary_model import create_binary_optimized_features

def create_simple_optimized_submission():
    # Load data and generate features
    train = pd.read_csv('/home/USER/Projects/RAIN/data/train.csv')
    test = pd.read_csv('/home/USER/Projects/RAIN/data/test.csv')
    X, X_test = create_binary_optimized_features(train, test)
    y = train['rainfall']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # Feature selection - try varying the threshold
    thresholds = ['mean', '0.8*mean', '1.2*mean', 'median']
    best_auc = 0
    best_threshold = None
    best_features = None
    
    for threshold in thresholds:
        # Base model for feature importance
        base_lr = LogisticRegression(C=0.5, max_iter=10000, solver='liblinear', random_state=42)
        
        # Select features
        selector = SelectFromModel(base_lr, threshold=threshold)
        selector.fit(X_scaled, y)
        selected = X_scaled.columns[selector.get_support()]
        
        if len(selected) < 5:  # Too few features
            continue
            
        X_selected = X_scaled[selected]
        
        # Cross-validate with these features
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        test_lr = LogisticRegression(C=0.5, max_iter=10000, solver='liblinear', random_state=42)
        preds = cross_val_predict(test_lr, X_selected, y, cv=kf, method='predict_proba')[:,1]
        auc = roc_auc_score(y, preds)
        
        print(f"Threshold {threshold}: {len(selected)} features, AUC {auc:.6f}")
        
        if auc > best_auc:
            best_auc = auc
            best_threshold = threshold
            best_features = selected
    
    print(f"Best threshold: {best_threshold}, features: {len(best_features)}, AUC: {best_auc:.6f}")
    
    # Fine-tune C value around 0.5
    C_values = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
    best_C = 0.5
    best_C_auc = 0
    
    X_best = X_scaled[best_features]
    X_test_best = X_test_scaled[best_features]
    
    for C in C_values:
        test_lr = LogisticRegression(C=C, max_iter=10000, solver='liblinear', random_state=42)
        preds = cross_val_predict(test_lr, X_best, y, cv=kf, method='predict_proba')[:,1]
        auc = roc_auc_score(y, preds)
        print(f"C value {C}: AUC {auc:.6f}")
        
        if auc > best_C_auc:
            best_C_auc = auc
            best_C = C
    
    print(f"Best C value: {best_C}, AUC: {best_C_auc:.6f}")
    
    # Train final model
    final_model = LogisticRegression(C=best_C, max_iter=10000, solver='liblinear', random_state=42)
    final_model.fit(X_best, y)
    
    # Generate predictions
    test_proba = final_model.predict_proba(X_test_best)[:,1]
    
    # Create submission
    submission = pd.DataFrame({
        'id': test['id'],
        'rainfall': test_proba
    })
    
    output_path = f"data/candidates/optimized_simple_auc_{best_C_auc:.6f}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    # Display feature importances
    importance = pd.DataFrame({
        'feature': best_features,
        'coefficient': final_model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print("\nTop features:")
    print(importance.to_string(index=False))

if __name__ == "__main__":
    create_simple_optimized_submission()