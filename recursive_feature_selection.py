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

def recursive_feature_selection():
    """Two-phase feature selection: elimination followed by selective re-addition"""
    # Load data
    train = pd.read_csv('/home/USER/Projects/RAIN/data/train.csv')
    test = pd.read_csv('/home/USER/Projects/RAIN/data/test.csv')

    # Custom feature engineering
    def FE(df):
        df = df.copy()
        df['cloudtest'] = (df.cloud==88).astype(int)
        return df
    
    train = FE(train)
    test = FE(test)

    # Generate features
    X, X_test = create_binary_optimized_features(train, test)
    y = train['rainfall']

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
    
    print("\n=== PHASE 1: FEATURE ELIMINATION ===")
    
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
    
    elimination_auc = best_auc
    elimination_features = best_features.copy()
    
    print("\n=== PHASE 1 RESULTS ===")
    print(f"Original AUC with {len(X.columns)} features: {baseline_auc:.6f}")
    print(f"After elimination AUC with {len(elimination_features)} features: {elimination_auc:.6f}")
    print(f"Improvement: {elimination_auc - baseline_auc:.6f}")
    print(f"Eliminated {len(eliminated_features)} features")
    
    # PHASE 2: Try adding back eliminated features
    print("\n=== PHASE 2: FEATURE RE-ADDITION ===")
    readded_features = []
    
    # Try adding back each eliminated feature
    for feature in eliminated_features:
        print(f"Testing re-addition of feature: {feature}")
        
        # Current feature set plus the candidate feature
        current_features = best_features + [feature]
        
        X_expanded = X[current_features]
        X_expanded_processed = preprocessing.fit_transform(X_expanded)
        
        scores = cross_val_score(lr, X_expanded_processed, y, cv=kf, scoring='roc_auc')
        current_auc = np.mean(scores)
        
        result = {
            'feature': feature,
            'phase': 'addition',
            'auc_before': best_auc,
            'auc_after': current_auc,
            'change': current_auc - best_auc,
            'action': 'skipped'
        }
        
        # If performance improves, add the feature back
        if current_auc > best_auc:
            best_features.append(feature)
            readded_features.append(feature)
            best_auc = current_auc
            result['action'] = 'added'
            print(f"✓ Re-added feature {feature}, AUC: {current_auc:.6f} (improvement: {current_auc - best_auc:.6f})")
        else:
            print(f"✗ Skipping feature {feature}, AUC would be {current_auc:.6f} (change: {current_auc - best_auc:.6f})")
        
        feature_results.append(result)
    
    final_auc = best_auc
    
    print("\n=== FINAL RESULTS ===")
    print(f"Original AUC with {len(X.columns)} features: {baseline_auc:.6f}")
    print(f"After elimination AUC with {len(elimination_features)} features: {elimination_auc:.6f}")
    print(f"Final AUC with {len(best_features)} features: {final_auc:.6f}")
    print(f"Total improvement: {final_auc - baseline_auc:.6f}")
    print(f"Eliminated {len(eliminated_features)} features, re-added {len(readded_features)} features")
    
    # Save results
    results_df = pd.DataFrame(feature_results)
    results_df.to_csv("data/candidates/feature_selection_results.csv", index=False)
    
    # Train final model
    lr_final = LogisticRegression(C=0.5, max_iter=10000, solver='liblinear', random_state=42)
    X_test_final = X_test[best_features]
    X_final_processed = preprocessing.fit_transform(X[best_features])
    X_test_final_processed = preprocessing.transform(X_test_final)
    
    lr_final.fit(X_final_processed, y)
    test_preds = lr_final.predict_proba(X_test_final_processed)[:,1]
    
    # Create submission
    submission = pd.DataFrame({
        'id': test['id'],
        'rainfall': test_preds
    })
    
    output_path = f"data/candidates/recursive_features_select_{final_auc:.6f}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    # Save final feature list
    pd.DataFrame({'feature': best_features}).to_csv(
        "data/candidates/optimal_feature_set.csv", index=False)
    
    # Detailed feature changes
    print("\nEliminated features (not added back):")
    for feature in eliminated_features:
        if feature not in readded_features:
            print(f"- {feature}")
    
    if readded_features:
        print("\nRe-added features:")
        for feature in readded_features:
            print(f"+ {feature}")
    
    return best_features, final_auc

if __name__ == "__main__":
    recursive_feature_selection()