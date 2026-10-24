import pandas as pd
import numpy as np
import os
import glob
import collections

def debug_submission_ids():
    """Debug the issue with submission IDs"""
    print("===== DEBUGGING SUBMISSION ID ISSUES =====\n")
    
    # Load all submission files from keep directory
    keep_dir = 'data/keep'
    submission_files = glob.glob(os.path.join(keep_dir, 'submission_*.csv'))
    print(f"Found {len(submission_files)} submission files in {keep_dir}")
    
    # Collect information about each submission
    submissions = []
    all_ids = collections.Counter()
    id_types = {}
    
    for file_path in submission_files:
        model_name = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)
            if 'id' not in df.columns or 'rainfall' not in df.columns:
                print(f"⚠️ {model_name}: Missing required columns")
                continue
                
            # Record basic info and sample IDs
            sub_info = {
                'name': model_name,
                'count': len(df),
                'id_type': type(df['id'].iloc[0]).__name__,
                'id_min': df['id'].min(),
                'id_max': df['id'].max(),
                'rainfall_min': df['rainfall'].min(),
                'rainfall_max': df['rainfall'].max(),
                'sample_ids': df['id'].head(5).tolist()
            }
            
            # Check for sortedness
            is_sorted = all(df['id'].iloc[i] <= df['id'].iloc[i + 1] for i in range(len(df) - 1))
            sub_info['is_sorted'] = is_sorted
            
            # Add to counter
            all_ids.update(df['id'].astype(str).tolist())
            
            # Track ID type
            type_name = type(df['id'].iloc[0]).__name__
            if type_name not in id_types:
                id_types[type_name] = []
            id_types[type_name].append(model_name)
            
            submissions.append(sub_info)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    # Check results
    if not submissions:
        print("No valid submissions found!")
        return
    
    # Print submission info
    print("\n===== SUBMISSION OVERVIEW =====")
    for sub in submissions[:5]:  # Show first 5
        print(f"\n{sub['name']}")
        print(f"  Records: {sub['count']}")
        print(f"  ID Type: {sub['id_type']}")
        print(f"  ID Range: {sub['id_min']} to {sub['id_max']}")
        print(f"  Sample IDs: {sub['sample_ids']}")
        print(f"  Sorted: {'Yes' if sub['is_sorted'] else 'No'}")
        print(f"  Rainfall Range: {sub['rainfall_min']:.4f} to {sub['rainfall_max']:.4f}")
    
    if len(submissions) > 5:
        print(f"\n... and {len(submissions) - 5} more submissions")
    
    # Summarize ID types
    print("\n===== ID TYPE SUMMARY =====")
    for type_name, files in id_types.items():
        print(f"Type {type_name}: {len(files)} files")
    
    # ID overlap analysis
    if len(submissions) > 1:
        print("\n===== ID OVERLAP ANALYSIS =====")
        
        # Find all unique IDs
        all_unique_ids = set()
        for sub in submissions:
            df = pd.read_csv(os.path.join(keep_dir, sub['name']))
            all_unique_ids.update(df['id'].astype(str).tolist())
        
        print(f"Total unique IDs across all submissions: {len(all_unique_ids)}")
        
        # Find IDs that appear in at least X% of submissions
        for pct in [100, 90, 75, 50, 25]:
            threshold = int(len(submissions) * pct / 100)
            common_ids = [id for id, count in all_ids.items() if count >= threshold]
            print(f"IDs in at least {pct}% of submissions ({threshold} files): {len(common_ids)}")
    
    # Look for potential test.csv file to compare
    print("\n===== TEST FILE REFERENCE =====")
    test_paths = ['data/test.csv', 'data/test_features.csv', 'data/test_processed.csv']
    test_file = None
    
    for path in test_paths:
        if os.path.exists(path):
            test_file = path
            test_df = pd.read_csv(path)
            print(f"Found {path} with {len(test_df)} records")
            print(f"Test ID Type: {type(test_df['id'].iloc[0]).__name__}")
            print(f"Test ID Range: {test_df['id'].min()} to {test_df['id'].max()}")
            print(f"Sample Test IDs: {test_df['id'].head(5).tolist()}")
            
            # Check if any submission IDs match test IDs
            for sub in submissions[:5]:
                sub_df = pd.read_csv(os.path.join(keep_dir, sub['name']))
                overlap = set(sub_df['id'].astype(str)) & set(test_df['id'].astype(str))
                print(f"ID overlap with {sub['name']}: {len(overlap)} IDs ({len(overlap)/len(test_df)*100:.1f}%)")
            
            break
    
    if not test_file:
        print("No test file found for reference")
    
    # Recommendation
    print("\n===== DIAGNOSIS & SOLUTION =====")
    
    if len(all_unique_ids) > 0 and len([id for id, count in all_ids.items() if count > 1]) == 0:
        print("CRITICAL ISSUE: No ID appears in more than one submission file!")
        print("This suggests the IDs are inconsistent or transformed differently in each file.")
        
        # Check if IDs look like strings vs numbers
        sample_ids = []
        for sub in submissions[:3]:
            df = pd.read_csv(os.path.join(keep_dir, sub['name']))
            sample_ids.extend(df['id'].astype(str).head(5).tolist())
        
        if any(not id.isdigit() for id in sample_ids):
            print("Some IDs are not numeric - they may contain text or special characters")
        
        # Solution recommendations
        print("\nRECOMMENDED SOLUTIONS:")
        print("1. Create a consistent mapping of IDs by running fix_submission_ids.py.")
        print("2. Manually inspect submission files to ensure IDs align with your test data.")
        print("3. Consider regenerating submissions with consistent ID handling.")
    else:
        print(f"Found {len([id for id, count in all_ids.items() if count > 1])} IDs that appear in multiple files.")
        print("Run advanced_ensemble_fixed.py to use a more robust ID handling approach.")

def fix_inconsistent_ids():
    """Create a version of each submission with normalized IDs"""
    print("===== FIXING INCONSISTENT SUBMISSION IDs =====\n")
    
    # Try to load test file as reference for correct IDs
    test_file = None
    for path in ['data/test.csv', 'data/test_features.csv', 'data/test_processed.csv']:
        if os.path.exists(path):
            test_file = path
            break
    
    if not test_file:
        print("No test file found! Cannot normalize IDs without a reference.")
        return
        
    # Load test IDs as the source of truth
    test_df = pd.read_csv(test_file)
    test_ids = test_df['id'].astype(str).tolist()
    print(f"Loaded {len(test_ids)} reference IDs from {test_file}")
    
    # Create directory for fixed submissions
    fixed_dir = 'data/submissions_fixed'
    os.makedirs(fixed_dir, exist_ok=True)
    
    # Find all submission files
    keep_dir = 'data/keep'
    submission_files = glob.glob(os.path.join(keep_dir, 'submission_*.csv'))
    print(f"Found {len(submission_files)} submission files to fix")
    
    for file_path in submission_files:
        file_name = os.path.basename(file_path)
        
        try:
            # Load submission
            sub_df = pd.read_csv(file_path)
            
            # Create a DataFrame with test IDs
            fixed_df = pd.DataFrame({'id': test_ids})
            
            # Convert submission IDs to strings for matching
            sub_df['id'] = sub_df['id'].astype(str)
            
            # Merge to get rainfall predictions that match test IDs
            fixed_df = fixed_df.merge(sub_df[['id', 'rainfall']], on='id', how='left')
            
            # Fill missing values with median of available values
            if fixed_df['rainfall'].isna().any():
                median_value = sub_df['rainfall'].median()
                fixed_df['rainfall'].fillna(median_value, inplace=True)
                print(f"{file_name}: Filled {fixed_df['rainfall'].isna().sum()} missing values with median {median_value:.4f}")
            
            # Save fixed submission
            fixed_path = os.path.join(fixed_dir, file_name)
            fixed_df.to_csv(fixed_path, index=False)
            print(f"Fixed {file_name} saved to {fixed_path}")
            
            # Also copy fixed version to keep directory with _fixed suffix
            fixed_keep_path = os.path.join(keep_dir, file_name.replace('.csv', '_fixed.csv'))
            fixed_df.to_csv(fixed_keep_path, index=False)
            
        except Exception as e:
            print(f"Error fixing {file_path}: {str(e)}")
    
    print("\nCreated fixed versions of all submissions with consistent IDs")
    print("You can now run advanced_ensemble_fixed.py with the fixed submissions")

if __name__ == "__main__":
    # First debug to understand the problem
    debug_submission_ids()
    
    # Ask if user wants to proceed with fixing IDs
    print("\nDo you want to proceed with fixing the ID inconsistencies? (y/n)")
    response = input().strip().lower()
    
    if response == 'y':
        fix_inconsistent_ids()
    else:
        print("Fix canceled. You can run this script again later if needed.")
