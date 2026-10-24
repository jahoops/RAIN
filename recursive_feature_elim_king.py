import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os
from optimized_binary_model import create_binary_optimized_features

def recursive_feature_elimination():
    """Recursively eliminate features that don't contribute to model performance"""
    # Load data
    train = pd.read_csv('/home/USER/Projects/RAIN/data/train.csv')
    test = pd.read_csv('/home/USER/Projects/RAIN/data/test.csv')

    def FE(df):
        df = df.copy()
        df['cloudtest'] =  (df.cloud==88).astype(int)
        return df
    
    train = FE(train)
    test = FE(test)

    # Generate features
    X, X_test = create_binary_optimized_features(train, test)
    y = train['rainfall']

    def FEDrop(df):
        df = df.copy()
        df = df.drop(['winddirection', 'day_sin', 'day_div_pressure', 'day_x_pressure', 'windspeed_sqrt', 'day_of_year', 'humidity', 'pressure_squared', 'pressure', 'cloud_squared', 'humidity_squared', 'maxtemp', 'temparature', 'day_x_maxtemp', 'day_div_temparature', 'pressure_x_maxtemp', 'pressure_x_temparature', 'pressure_div_temparature', 'maxtemp_x_temparature', 'maxtemp_div_temparature'], axis=1)
        return df
    
    X = FEDrop(X)
    X_test = FEDrop(X_test)    

    # Apply SMOTE for class balance
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    
    # Create preprocessing pipeline
    preprocessing = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Base model
    lr = LogisticRegression(C=0.5, max_iter=10000, solver='liblinear', random_state=42)
    
    # Track feature elimination
    features_to_keep = list(X.columns)
    eliminated_features = []
    
    # Get baseline performance
    X_processed = preprocessing.fit_transform(X)
    baseline_scores = cross_val_score(lr, X_processed, y, cv=kf, scoring='roc_auc')
    baseline_auc = np.mean(baseline_scores)
    print(f"Baseline AUC with all {len(features_to_keep)} features: {baseline_auc:.6f}")
    
    best_auc = baseline_auc
    best_features = features_to_keep.copy()
    
    feature_results = []
    
    # Load feature importance (optional)
    try:
        importance_df = pd.read_csv('/home/USER/Projects/RAIN/data/candidates/lr_feature_importance.csv')
        # Sort features by importance (ascending so we test least important first)
        sorted_features = importance_df.sort_values('coefficient', key=abs).feature.tolist()
        # Ensure all features in X are in the sorted list
        for feature in features_to_keep:
            if feature not in sorted_features:
                sorted_features.append(feature)
    except:
        # If feature importance file not available, use alphabetical order
        sorted_features = sorted(features_to_keep)
    
    # Iterate through features in order of importance (least to most)
    for feature in sorted_features:
        if feature not in features_to_keep:
            continue
            
        print(f"Testing removal of feature: {feature}")
        
        # Remove this feature
        current_features = [f for f in features_to_keep if f != feature]
        
        # Skip if no features left (shouldn't happen in practice)
        if not current_features:
            continue
        
        # Evaluate model without this feature
        X_reduced = X[current_features]
        X_reduced_processed = preprocessing.fit_transform(X_reduced)
        
        scores = cross_val_score(lr, X_reduced_processed, y, cv=kf, scoring='roc_auc')
        current_auc = np.mean(scores)
        
        result = {
            'feature': feature,
            'auc_before': best_auc,
            'auc_after': current_auc,
            'change': current_auc - best_auc,
            'action': 'kept'
        }
        
        # If performance same or improves, permanently remove the feature
        if current_auc >= best_auc:
            features_to_keep.remove(feature)
            eliminated_features.append(feature)
            best_auc = current_auc
            best_features = current_features
            result['action'] = 'removed'
            print(f"✓ Removed feature {feature}, AUC: {current_auc:.6f} (change: {current_auc - best_auc:.6f})")
        else:
            print(f"✗ Keeping feature {feature}, AUC would decrease to {current_auc:.6f} (change: {current_auc - best_auc:.6f})")
        
        feature_results.append(result)
    
    # Final evaluation with best feature set
    X_final = X[best_features]
    X_final_processed = preprocessing.fit_transform(X_final)
    final_scores = cross_val_score(lr, X_final_processed, y, cv=kf, scoring='roc_auc')
    final_auc = np.mean(final_scores)
    
    print("\n=== Final Results ===")
    print(f"Original AUC with {len(X.columns)} features: {baseline_auc:.6f}")
    print(f"Final AUC with {len(best_features)} features: {final_auc:.6f}")
    print(f"Improvement: {final_auc - baseline_auc:.6f}")
    print(f"Eliminated {len(eliminated_features)} features")

    print(eliminated_features)
    
    # Save results
    results_df = pd.DataFrame(feature_results)
    results_df.to_csv("data/candidates/feature_elimination_results.csv", index=False)
    
    # Train final model
    lr_final = LogisticRegression(C=0.5, max_iter=10000, solver='liblinear', random_state=42)
    X_test_final = X_test[best_features]
    X_final_processed = preprocessing.fit_transform(X_final)
    X_test_final_processed = preprocessing.transform(X_test_final)
    
    lr_final.fit(X_final_processed, y)
    test_preds = lr_final.predict_proba(X_test_final_processed)[:,1]
    
    # Create submission
    submission = pd.DataFrame({
        'id': test['id'],
        'rainfall': test_preds
    })
    
    output_path = f"data/candidates/optimal_features_auc_{final_auc:.6f}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    # Save final feature list
    pd.DataFrame({'feature': best_features}).to_csv(
        "data/candidates/optimal_feature_set.csv", index=False)
    print("\nRemoved features:")
    for feature in eliminated_features:
        print(f"- {feature}")
    
    return best_features, final_auc

if __name__ == "__main__":
    recursive_feature_elimination()