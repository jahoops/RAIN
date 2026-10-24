import os
import pandas as pd
import numpy as np

def fix_data_bug():
    """Fix the critical bug in load_data_with_consistency function"""
    print("===== FIXING CRITICAL DATA LOADING BUG =====")
    
    # Open winning_blend.py and find the bug
    with open('winning_blend.py', 'r') as f:
        code = f.read()
    
    # Check if the bug is present
    if 'X_test = X[common_cols]' in code:
        print("FOUND CRITICAL BUG: X_test is being set to a subset of X instead of X_test!")
        print("This causes X_test to have 730 rows (same as train) instead of 2190 rows (test)")
        print("The test_ids still has 2190 elements, causing the length mismatch error\n")
        
        # Create fixed code
        fixed_code = code.replace(
            'X_test = X[common_cols]', 
            'X_test = X_test[common_cols]  # Fixed: use X_test instead of X'
        )
        
        # Save backup of original
        with open('winning_blend.py.bak', 'w') as f:
            f.write(code)
        print("Original winning_blend.py backed up to winning_blend.py.bak")
        
        # Save fixed version
        with open('winning_blend.py', 'w') as f:
            f.write(fixed_code)
        print("Fixed winning_blend.py saved")
        
        print("\nThe fix corrects a copy-paste error in load_data_with_consistency:")
        print("Original (broken): X_test = X[common_cols]")
        print("Fixed: X_test = X_test[common_cols]")
        
        print("\nThis bug was accidentally assigning train data to test,")
        print("causing a mismatch between test_ids (2190 rows) and X_test (730 rows).")
    else:
        print("The specific bug wasn't found in the code.")
        print("As an alternative, add these checks to train_diverse_models:")
        
        print("""
        # CRITICAL FIX: Make sure test_ids match X_test rows
        if len(test_ids) != len(X_test):
            print(f"WARNING: Length mismatch detected! test_ids: {len(test_ids)}, X_test: {len(X_test)}")
            # If X_test has been filtered/reduced, we need to align test_ids to match
            test_ids = test_ids[:len(X_test)]
            print(f"Trimmed test_ids to match X_test length: {len(test_ids)}")
        """)
    
    print("\nAfter fixing, run winning_blend.py again.")

if __name__ == "__main__":
    fix_data_bug()
