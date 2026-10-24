import os
import pandas as pd
import numpy as np

def fix_winning_blend():
    """
    Fix the array length mismatch in winning_blend.py
    """
    print("===== FIXING TEST IDS AND FEATURES MISMATCH =====")
    
    # First, let's confirm if data files exist to see what's happening
    test_paths = [
        'data/test.csv',
        'data/test_processed.csv',
        'data/test_features.csv'
    ]
    
    print("Checking available test data files...")
    test_file = None
    for path in test_paths:
        if os.path.exists(path):
            test_file = path
            print(f"Found test file: {path}")
    
    if test_file:
        # Load the original test data to get proper test_ids
        test_df = pd.read_csv(test_file)
        test_ids = test_df['id'].values
        print(f"Original test dataset has {len(test_ids)} samples")
    
    # Now let's examine if any model files exist
    print("\nChecking for model files...")
    model_dirs = ['models', 'models/diverse']
    for dir_path in model_dirs:
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            model_files = [f for f in files if f.endswith('.pkl') or f.endswith('.cbm')]
            if model_files:
                print(f"Found {len(model_files)} model files in {dir_path}")
    
    print("\nHere's the fix for winning_blend.py:")
    print("The error 'array length 2190 does not match index length 730' indicates")
    print("that your test_ids and X_test_selected have different lengths.")
    print("\nAdd this code to the beginning of the train_diverse_models function:")
    
    print("""
    def train_diverse_models(X: pd.DataFrame, y: pd.DataFrame, X_test: pd.DataFrame, 
                            test_ids: pd.Series) -> Dict:
        \"\"\"Train diverse models with different algorithms and hyperparameters\"\"\"
        print("Training diverse models...")
        
        # CRITICAL FIX: Make sure test_ids match X_test rows
        if len(test_ids) != len(X_test):
            print(f"WARNING: Length mismatch detected! test_ids: {len(test_ids)}, X_test: {len(X_test)}")
            # If X_test has been filtered/reduced, we need to align test_ids to match
            if isinstance(test_ids, pd.Series) and 'id' in X_test.columns:
                # Use only test_ids that exist in X_test
                test_ids = test_ids[test_ids.isin(X_test['id'])]
                print(f"Aligned test_ids to match X_test, new length: {len(test_ids)}")
            else:
                # If no ID column in X_test, we have to assume same order but different length
                # This is a fallback that might not be accurate!
                print("WARNING: No id column in X_test, using first portion of test_ids")
                test_ids = test_ids[:len(X_test)]
        
        print(f"Final dimensions - X: {X.shape}, y: {len(y)}, X_test: {X_test.shape}, test_ids: {len(test_ids)}")
    """)
    
    print("\nAlternative fix: Modify load_data_with_consistency to ensure test IDs and features remain aligned:")
    print("""
    def load_data_with_consistency():
        # ... existing code ...
        
        # After all preprocessing:
        # Ensure test_ids match the rows in X_test
        test_ids_aligned = test_ids.copy()  # Make a copy of original IDs
        
        # If we have a reference to map between original and filtered test data
        if 'id' in X_test.columns:
            test_ids_aligned = X_test['id'].values
            X_test = X_test.drop(['id'], axis=1)
        else:
            # If no ID column, assume test_ids and X_test are in the same order
            # but test_ids might need truncating if X_test has fewer rows
            if len(test_ids_aligned) != X_test.shape[0]:
                test_ids_aligned = test_ids_aligned[:X_test.shape[0]]
        
        print(f"Final data shapes: X={X.shape}, y={len(y)}, X_test={X_test.shape}, test_ids={len(test_ids_aligned)}")
        return X, y, X_test, train_ids, test_ids_aligned
    """)
    
    # Check for any existing submissions that might provide clues
    print("\nChecking existing submissions...")
    submission_dir = 'data'
    if os.path.exists(submission_dir):
        submissions = [f for f in os.listdir(submission_dir) if f.startswith('submission_') and f.endswith('.csv')]
        if submissions:
            print(f"Found {len(submissions)} submission files:")
            for sub in submissions[:3]:  # Show first 3
                try:
                    sub_df = pd.read_csv(os.path.join(submission_dir, sub))
                    print(f"  - {sub}: {len(sub_df)} rows")
                except:
                    print(f"  - {sub}: Could not read file")
            if len(submissions) > 3:
                print(f"  ... and {len(submissions) - 3} more")

if __name__ == "__main__":
    fix_winning_blend()
