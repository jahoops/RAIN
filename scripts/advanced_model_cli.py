import os
import pandas as pd
import numpy as np
from fix_advanced_model import (load_and_prepare_data, 
                               create_consistent_features, 
                               train_ensemble_model)
import argparse

def run_advanced_model():
    print("\n===== RAIN Advanced Model Training (Fixed Version) =====\n")
    print("1. Loading and preprocessing data...")
    X, y, X_test, train_ids, test_ids = load_and_prepare_data()
    
    print("\n2. Creating consistent features for train and test...")
    X_enhanced, X_test_enhanced = create_consistent_features(X, X_test)
    
    # Check feature consistency
    if X_enhanced.shape[1] != X_test_enhanced.shape[1]:
        print(f"ERROR: Feature count mismatch! Train: {X_enhanced.shape[1]}, Test: {X_test_enhanced.shape[1]}")
        print("Attempting to fix by using only common features...")
        common_cols = list(set(X_enhanced.columns).intersection(set(X_test_enhanced.columns)))
        X_enhanced = X_enhanced[common_cols]
        X_test_enhanced = X_test_enhanced[common_cols]
        print(f"Using {len(common_cols)} common features")
    else:
        print(f"Feature consistency verified: both train and test have {X_enhanced.shape[1]} features")
    
    for col1, col2 in zip(X_enhanced.columns, X_test_enhanced.columns):
        if col1 != col2:
            print(f"ERROR: Column order mismatch at position {list(X_enhanced.columns).index(col1)}")
            print(f"Train: {col1}, Test: {col2}")
            break
    
    print("\n3. Training ensemble model...")
    results = train_ensemble_model(X_enhanced, y, X_test_enhanced, test_ids)
    
    print("\n4. Running submission validation...")
    try:
        from validate_submission import validate_submission
        for sub_file in ['data/submission_ensemble_fixed.csv', 
                        'data/submission_xgb_fixed.csv',
                        'data/submission_lgb_fixed.csv']:
            print(f"\nValidating {sub_file}:")
            validate_submission(sub_file, quiet=False)
    except Exception as e:
        print(f"Validation error: {str(e)}")

    print("\n5. Creating backup of successful models...")
    try:
        # Copy submissions to keep directory
        os.makedirs('data/keep', exist_ok=True)
        import shutil
        for file in ['submission_ensemble_fixed.csv', 'submission_xgb_fixed.csv', 'submission_lgb_fixed.csv']:
            source = os.path.join('data', file)
            dest = os.path.join('data/keep', file)
            shutil.copy2(source, dest)
            print(f"Saved {file} to keep directory")
    except Exception as e:
        print(f"Error creating backups: {str(e)}")
            
    print("\nAdvanced model training complete!")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced RAIN Model Training")
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    args = parser.parse_args()
    
    run_advanced_model()
