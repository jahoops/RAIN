import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel
import os
from optimized_binary_model import create_binary_optimized_features

def explore_regularization():
    """Explore different regularization approaches"""
    # Load data
    train = pd.read_csv('/home/USER/Projects/RAIN/data/train.csv')
    test = pd.read_csv('/home/USER/Projects/RAIN/data/test.csv')
    
    # Generate features
    X, X_test = create_binary_optimized_features(train, test)
    y = train['rainfall']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # Feature selection
    base_lr = LogisticRegression(C=0.5, max_iter=10000, solver='liblinear', random_state=42)
    selector = SelectFromModel(base_lr, threshold='mean')
    selector.fit(X_scaled, y)
    selected_features = X_scaled.columns[selector.get_support()]
    X_selected = X_scaled[selected_features]
    X_test_selected = X_test_scaled[selected_features]
    
    print(f"Using {len(selected_features)} selected features")
    
    # Explore different regularization
    reg_types = [
        ('l2', 'liblinear'),  # Default
        ('l1', 'liblinear'),  # L1 regularization (feature selection)
        ('none', 'newton-cg')  # No regularization
    ]
    
    best_auc = 0
    best_config = None
    results = []
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for penalty, solver in reg_types:
        for C in [0.2, 0.5, 1.0, 2.0]:
            for class_weight in [None, 'balanced']:
                try:
                    print(f"Testing penalty={penalty}, solver={solver}, C={C}, class_weight={class_weight}")
                    
                    lr = LogisticRegression(
                        penalty=penalty,
                        C=C,
                        solver=solver,
                        class_weight=class_weight,
                        max_iter=10000,
                        random_state=42
                    )
                    
                    preds = cross_val_predict(lr, X_selected, y, cv=kf, method='predict_proba')[:,1]
                    auc = roc_auc_score(y, preds)
                    
                    results.append({
                        'penalty': penalty,
                        'solver': solver,
                        'C': C,
                        'class_weight': str(class_weight),
                        'auc': auc
                    })
                    
                    print(f"AUC: {auc:.6f}")
                    
                    if auc > best_auc:
                        best_auc = auc
                        best_config = {
                            'penalty': penalty,
                            'solver': solver,
                            'C': C,
                            'class_weight': class_weight
                        }
                except Exception as e:
                    print(f"Error: {e}")
    
    # Print results
    results_df = pd.DataFrame(results).sort_values('auc', ascending=False)
    print("\nTop regularization configurations:")
    print(results_df.head(10).to_string(index=False))
    
    # Save best model
    print(f"\nBest configuration: {best_config}, AUC: {best_auc:.6f}")
    
    # Train final model with best config
    best_lr = LogisticRegression(
        penalty=best_config['penalty'],
        C=best_config['C'],
        solver=best_config['solver'],
        class_weight=best_config['class_weight'],
        max_iter=10000,
        random_state=42
    )
    
    best_lr.fit(X_selected, y)
    test_proba = best_lr.predict_proba(X_test_selected)[:,1]
    
    # Create submission
    submission = pd.DataFrame({
        'id': test['id'],
        'rainfall': test_proba
    })
    
    output_path = f"data/candidates/best_reg_auc_{best_auc:.6f}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

if __name__ == "__main__":
    explore_regularization()