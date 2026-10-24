import pandas as pd
import numpy as np
import os

def fix_submission_nan():
    """Fix NaN values in ensemble submissions"""
    print("=== CHECKING AND FIXING NAN VALUES IN SUBMISSIONS ===")
    
    # Try to load the problematic submission
    submission_path = 'data/submission_winning_ensemble_fixed3.csv'
    
    if not os.path.exists(submission_path):
        print(f"File {submission_path} does not exist.")
        return
    
    df = pd.read_csv(submission_path)
    
    print(f"Loaded submission from {submission_path}")
    print(f"Shape: {df.shape}")
    
    # Check for NaN values
    nan_count = df['rainfall'].isna().sum()
    if nan_count > 0:
        print(f"Found {nan_count} NaN values in rainfall predictions")
        
        # Find a reasonable value to replace NaNs with
        if df['rainfall'].notna().any():
            # Use the median of valid values
            median_value = df['rainfall'].median()
            print(f"Replacing NaN values with median: {median_value:.4f}")
            df['rainfall'] = df['rainfall'].fillna(median_value)
        else:
            # If all values are NaN, use 0.5 as a default
            print("All values are NaN, replacing with default value 0.5")
            df['rainfall'] = 0.5
    else:
        print("No NaN values found in rainfall column")
    
    # Check for infinity values
    inf_count = np.isinf(df['rainfall']).sum()
    if inf_count > 0:
        print(f"Found {inf_count} infinity values in rainfall predictions")
        # Replace infinities with NaN first, then with median
        df['rainfall'] = df['rainfall'].replace([np.inf, -np.inf], np.nan)
        
        # Use the median of valid values, or 0.5 if none
        if df['rainfall'].notna().any():
            median_value = df['rainfall'].median()
        else:
            median_value = 0.5
            
        print(f"Replacing infinity values with {median_value:.4f}")
        df['rainfall'] = df['rainfall'].fillna(median_value)
    else:
        print("No infinity values found in rainfall column")
    
    # Check for range issues
    min_val = df['rainfall'].min()
    max_val = df['rainfall'].max()
    print(f"Value range: {min_val:.4f} to {max_val:.4f}")
    
    # Ensure values are in reasonable range for rainfall probabilities
    if min_val < 0 or max_val > 1:
        print("Values outside expected range [0,1], clipping to valid range")
        df['rainfall'] = np.clip(df['rainfall'], 0, 1)
    
    # Save fixed submission
    fixed_path = submission_path.replace('.csv', '_fixed.csv')
    df.to_csv(fixed_path, index=False)
    print(f"Saved fixed submission to {fixed_path}")
    
    # Also save a copy to keep directory
    keep_path = os.path.join('data/keep', os.path.basename(fixed_path))
    df.to_csv(keep_path, index=False)
    print(f"Saved copy to {keep_path}")
    
    # Now try to validate the fixed submission
    try:
        from validate_submission import validate_submission
        print("\nValidating fixed submission...")
        result = validate_submission(fixed_path)
        if result and 'estimated_score' in result:
            print(f"Estimated score: {result['estimated_score']:.6f}")
            return True
    except Exception as e:
        print(f"Could not validate fixed submission: {e}")
    
    # Also try to recover fixed id files directly
    try:
        # List all fixed id files 
        fixed_dir = 'data/fixed_submissions'
        if os.path.exists(fixed_dir):
            print("\nTrying direct ensemble from fixed submission files...")
            
            submission_files = [f for f in os.listdir(fixed_dir) if f.endswith('.csv')]
            if submission_files:
                print(f"Found {len(submission_files)} fixed submission files")
                
                # Load all files and aggregate
                all_dfs = []
                for file in submission_files[:10]:  # Limit to first 10 to avoid memory issues
                    df = pd.read_csv(os.path.join(fixed_dir, file))
                    if 'id' in df.columns and 'rainfall' in df.columns:
                        all_dfs.append(df)
                
                # Create simple average ensemble
                if all_dfs:
                    print(f"Creating ensemble from {len(all_dfs)} files")
                    
                    # Use first file as base and align IDs
                    base_df = all_dfs[0].copy()
                    base_df['rainfall'] = 0
                    
                    # Simple average of all predictions
                    for df in all_dfs:
                        # Add 1/n of each file's prediction
                        merged = base_df.merge(df, on='id', how='left', suffixes=('', '_other'))
                        base_df['rainfall'] += merged['rainfall_other'].fillna(merged['rainfall']) / len(all_dfs)
                    
                    # Check and fix any issues
                    base_df['rainfall'] = base_df['rainfall'].fillna(0.5)
                    base_df['rainfall'] = np.clip(base_df['rainfall'], 0, 1)
                    
                    # Save the ensemble
                    direct_path = 'data/submission_direct_ensemble.csv'
                    base_df.to_csv(direct_path, index=False)
                    print(f"Saved direct ensemble to {direct_path}")
                    
                    # Copy to keep
                    keep_direct = os.path.join('data/keep', 'submission_direct_ensemble.csv')
                    base_df.to_csv(keep_direct, index=False)
                    
                    # Validate
                    try:
                        result = validate_submission(direct_path)
                        if result and 'estimated_score' in result:
                            print(f"Direct ensemble score: {result['estimated_score']:.6f}")
                    except Exception as e:
                        print(f"Could not validate direct ensemble: {e}")
    except Exception as e:
        print(f"Error in direct ensemble creation: {e}")
    
    return False

def regenerate_from_u_b():
    """Generate a submission duplicating u_b for safety"""
    print("\n=== GENERATING SAFE SUBMISSION BASED ON U_B ===")
    
    u_b_path = 'data/keep/submission_u_b.csv'
    if not os.path.exists(u_b_path):
        print(f"Cannot find u_b submission at {u_b_path}")
        return False
    
    try:
        u_b_df = pd.read_csv(u_b_path)
        print(f"Loaded u_b submission with {len(u_b_df)} rows")
        
        # Make a simple copy as safe submission
        safe_path = 'data/submission_safe.csv'
        u_b_df.to_csv(safe_path, index=False)
        print(f"Saved safe submission to {safe_path}")
        
        # Check for a test reference to ensure IDs align with expected test set
        test_path = 'data/test.csv'
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            if len(test_df) != len(u_b_df):
                print(f"WARNING: u_b submission has {len(u_b_df)} rows, but test has {len(test_df)} rows")
            
            # Check if IDs match
            common_ids = set(u_b_df['id']).intersection(set(test_df['id']))
            print(f"u_b submission has {len(common_ids)} IDs in common with test file")
            
            if len(common_ids) < len(test_df):
                print("WARNING: Some test IDs are missing from the u_b submission")
                
                # Create a version aligned with test IDs
                aligned_df = pd.DataFrame({'id': test_df['id']})
                aligned_df = aligned_df.merge(u_b_df, on='id', how='left')
                aligned_df['rainfall'] = aligned_df['rainfall'].fillna(0.5)
                
                aligned_path = 'data/submission_aligned.csv'
                aligned_df[['id', 'rainfall']].to_csv(aligned_path, index=False)
                print(f"Saved test-aligned submission to {aligned_path}")
        
        # Validate the submission
        try:
            from validate_submission import validate_submission
            print("\nValidating safe submission...")
            result = validate_submission(safe_path)
            if result and 'estimated_score' in result:
                print(f"Estimated score: {result['estimated_score']:.6f}")
                return True
        except Exception as e:
            print(f"Could not validate safe submission: {e}")
        
        return True
    except Exception as e:
        print(f"Error creating safe submission: {e}")
        return False

if __name__ == "__main__":
    print("Attempting to fix issues in ensemble submission...")
    if not fix_submission_nan():
        print("\nCould not fix ensemble submission. Trying to create a safe submission instead...")
        regenerate_from_u_b()
