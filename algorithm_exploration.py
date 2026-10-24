import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import os
from optimized_binary_model import create_binary_optimized_features
from sklearn.feature_selection import SelectFromModel

def explore_classifiers():
    """Try different classification algorithms on the same features"""
    # Load data
    train = pd.read_csv('/home/USER/Projects/RAIN/data/train.csv')
    test = pd.read_csv('/home/USER/Projects/RAIN/data/test.csv')
    
    # Generate features - use the same feature set that worked well
    X, X_test = create_binary_optimized_features(train, test)
    y = train['rainfall']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # Feature selection - use the threshold that worked before
    base_lr = LogisticRegression(C=0.5, max_iter=10000, solver='liblinear', random_state=42)
    selector = SelectFromModel(base_lr, threshold='mean')
    selector.fit(X_scaled, y)
    selected_features = X_scaled.columns[selector.get_support()]
    X_selected = X_scaled[selected_features]
    X_test_selected = X_test_scaled[selected_features]
    
    print(f"Using {len(selected_features)} selected features")
    
    # Define classifiers to try
    classifiers = {
        'logistic': LogisticRegression(C=0.5, max_iter=10000, solver='liblinear', random_state=42),
        'sgd': SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, random_state=42),
        'svm': SVC(probability=True, C=1.0, kernel='rbf', gamma='scale', random_state=42),
        'rf': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        'knn': KNeighborsClassifier(n_neighbors=15, weights='distance'),
        'mlp': MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
    }
    
    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, clf in classifiers.items():
        try:
            print(f"Training {name}...")
            preds = cross_val_predict(clf, X_selected, y, cv=kf, method='predict_proba')[:,1]
            auc = roc_auc_score(y, preds)
            results[name] = auc
            print(f"{name} AUC: {auc:.6f}")
            
            # Train final model and create submission
            clf.fit(X_selected, y)
            test_proba = clf.predict_proba(X_test_selected)[:,1]
            
            submission = pd.DataFrame({
                'id': test['id'],
                'rainfall': test_proba
            })
            
            output_path = f"data/candidates/{name}_auc_{auc:.6f}.csv"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            submission.to_csv(output_path, index=False)
            print(f"Submission saved to {output_path}")
        except Exception as e:
            print(f"Error with {name}: {e}")
    
    # Print final comparison
    print("\nAlgorithm comparison:")
    for name, auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {auc:.6f}")

if __name__ == "__main__":
    explore_classifiers()