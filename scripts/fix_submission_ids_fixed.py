import pandas as pd
import numpy as np
import os
import glob

def analyze_id_mismatch():
    """Analyze the ID mismatch in submission files"""
    print("===== ANALYZING SUBMISSION ID MISMATCH =====")
    
    # Load submission files from keep directory
    keep_dir = 'data/keep'
    submission_files = glob.glob(os.path.join(keep_dir, 'submission_*.csv'))
    
    if not submission_files:
        print("No submission files found in keep directory")
        return None, None
    
    # Load submissions and analyze IDs
    submissions = {}
    id_types = {}
    id_ranges = {}
    id_samples = {}
    id_counts = {}
    
    for file_path in submission_files:
        model_name = os.path.basename(file_path).replace('submission_', '').replace('.csv', '')
        try:
            df = pd.read_csv(file_path)
            submissions[model_name] = df
            
            # Analyze ID column
            if 'id' in df.columns:
                id_col = df['id']
                id_type = type(id_col.iloc[0]).__name__ if len(id_col) > 0 else "unknown"
                id_types[model_name] = id_type
                id_ranges[model_name] = (id_col.min(), id_col.max())
                id_samples[model_name] = id_col.head(5).tolist()
                id_counts[model_name] = len(id_col)
                print(f"{model_name}: {len(df)} rows, ID type: {id_type}, Range: {id_col.min()}-{id_col.max()}")
            else:
                print(f"{model_name}: Missing ID column!")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Check for test.csv to get reference IDs
    test_df = None
    test_paths = ['data/test.csv', 'data/test_processed.csv', 'data/test_features.csv']
    for path in test_paths:
        if os.path.exists(path):
            test_df = pd.read_csv(path)
            print(f"Found reference test file: {path} with {len(test_df)} rows")
            print(f"Test ID type: {type(test_df['id'].iloc[0]).__name__}")
            print(f"Test ID range: {test_df['id'].min()}-{test_df['id'].max()}")
            print(f"Test ID samples: {test_df['id'].head(5).tolist()}")
            break
    
    # Fix: Check if test_df is None rather than using "if not test_df"
    if test_df is None:
        print("Could not find test file for reference")
        
    # Check if IDs are just different types but same values
    print("\nID TYPE ANALYSIS:")
    if len(set(id_types.values())) > 1:
        print("Submissions have different ID types - this may be causing the mismatch")
        for model, id_type in id_types.items():
            print(f"  {model}: {id_type}")
    else:
        print(f"All submissions have consistent ID type: {next(iter(id_types.values()))}")
    
    # Check for index vs actual ID issue
    print("\nINDEX VS ID ANALYSIS:")
    sequential_ids = False
    for model, samples in id_samples.items():
        if samples == list(range(samples[0], samples[0] + len(samples))):
            sequential_ids = True
            print(f"  {model} has sequential IDs - may be using row index instead of true IDs")
    
    # Determine fix strategy
    print("\n===== FIX STRATEGY =====")
    if test_df is not None:
        print("Will use test.csv IDs as reference for fixing all submissions")
        reference_ids = test_df['id'].values
    else:
        # If no test file, use the u_b submission if available, or first submission otherwise
        reference_model = 'u_b' if 'u_b' in submissions else list(submissions.keys())[0]
        print(f"No test file found - using {reference_model} as reference for IDs")
        reference_ids = submissions[reference_model]['id'].values
    
    return submissions, reference_ids

# The rest of the code remains the same
def fix_submission_ids(submissions, reference_ids):
    """Fix ID issues in submission files"""
    print("\n===== FIXING SUBMISSION IDS =====")
    
    # Create output directory
    fixed_dir = 'data/fixed_submissions'
    os.makedirs(fixed_dir, exist_ok=True)
    
    # Fix each submission
    for model_name, df in submissions.items():
        print(f"Fixing {model_name}...")
        
        # Create a new dataframe with reference IDs
        fixed_df = pd.DataFrame({'id': reference_ids})
        
        # If the submission has same length as reference IDs but different values,
        # assume they're in the same order
        if len(df) == len(reference_ids) and not all(df['id'] == reference_ids):
            print(f"  {model_name} has same row count but different IDs - assuming same order")
            fixed_df['rainfall'] = df['rainfall'].values
        else:
            # Try to merge by ID first
            merged = fixed_df.merge(df, on='id', how='left')
            if 'rainfall' in merged.columns:
                fixed_df['rainfall'] = merged['rainfall']
                missing = merged['rainfall'].isna().sum()
                if missing > 0:
                    print(f"  {model_name} is missing {missing} IDs after merge - filling with median")
                    fixed_df['rainfall'] = fixed_df['rainfall'].fillna(df['rainfall'].median())
            else:
                # If merge doesn't work, try to align by index order
                print(f"  {model_name} IDs don't match reference - using index-based alignment")
                if len(df) >= len(reference_ids):
                    fixed_df['rainfall'] = df['rainfall'].values[:len(reference_ids)]
                else:
                    # If submission is shorter, pad with median
                    fixed_df['rainfall'] = np.nan
                    fixed_df.loc[:len(df)-1, 'rainfall'] = df['rainfall'].values
                    fixed_df['rainfall'] = fixed_df['rainfall'].fillna(df['rainfall'].median())
        
        # Save fixed submission
        fixed_path = os.path.join(fixed_dir, f'submission_{model_name}_fixed.csv')
        fixed_df.to_csv(fixed_path, index=False)
        print(f"  Saved fixed submission to {fixed_path}")
        
        # Also create a copy in the keep directory
        keep_path = os.path.join('data/keep', f'submission_{model_name}_fixed.csv')
        fixed_df.to_csv(keep_path, index=False)
    
    print("\nAll submissions fixed. Run winning_ensemble_fixed.py to create the ensemble.")

def create_fixed_winning_ensemble():
    """Create a winning ensemble from fixed submissions"""
    print("\n===== CREATING FIXED WINNING ENSEMBLE =====")
    
    # Load all fixed submissions
    fixed_dir = 'data/fixed_submissions'
    submission_files = glob.glob(os.path.join(fixed_dir, '*.csv'))
    
    if not submission_files:
        print("No fixed submission files found")
        return
    
    # Load submissions
    submissions = {}
    for file_path in submission_files:
        model_name = os.path.basename(file_path).replace('submission_', '').replace('_fixed.csv', '')
        df = pd.read_csv(file_path)
        submissions[model_name] = df
        print(f"Loaded fixed {model_name} submission: {len(df)} rows")
    
    # Get the common IDs (should be all the same now)
    reference_ids = submissions[list(submissions.keys())[0]]['id'].values
    print(f"Using {len(reference_ids)} common IDs")
    
    # Create ensemble dataframe
    ensemble_df = pd.DataFrame({'id': reference_ids})
    
    # Add each model's predictions
    for name, df in submissions.items():
        ensemble_df[name] = df['rainfall'].values
    
    # Define weights (prioritize neural and synthetic_neural)
    weights = {}
    priority_models = ['synthetic_neural', 'neural', 'u_b', 'uncertainty_model']
    
    # Define weights based on model priority
    total_weight = 1.0
    priority_weight = 0.7
    priority_count = sum(1 for m in priority_models if m in submissions)
    
    if priority_count > 0:
        # Distribute priority weight among priority models
        for model in priority_models:
            if model in submissions:
                weights[model] = priority_weight / priority_count
        
        # Distribute remaining weight among other models
        remaining_models = [m for m in submissions.keys() if m not in priority_models]
        remaining_weight = total_weight - priority_weight
        
        if remaining_models:
            for model in remaining_models:
                weights[model] = remaining_weight / len(remaining_models)
    else:
        # Equal weights if no priority models
        for model in submissions:
            weights[model] = 1.0 / len(submissions)
    
    # Print weights
    print("\nEnsemble weights:")
    for model, weight in weights.items():
        print(f"{model}: {weight:.4f}")
    
    # Create weighted ensemble
    ensemble_df['winning_blend'] = 0
    for model, weight in weights.items():
        ensemble_df['winning_blend'] += ensemble_df[model] * weight
    
    # Create the submission
    submission = pd.DataFrame({
        'id': ensemble_df['id'],
        'rainfall': ensemble_df['winning_blend']
    })
    
    # Save submission
    output_path = 'data/submission_winning_ensemble_fixed.csv'
    submission.to_csv(output_path, index=False)
    print(f"\nFixed winning ensemble saved to {output_path}")
    
    # Also save in keep directory
    keep_path = 'data/keep/submission_winning_ensemble_fixed.csv'
    submission.to_csv(keep_path, index=False)
    print(f"Copy saved to {keep_path}")
    
    # Validate submission if possible
    try:
        from validate_submission import validate_submission
        print("\nValidating fixed ensemble...")
        result = validate_submission(output_path)
        if result and 'estimated_score' in result:
            print(f"Estimated score: {result['estimated_score']:.6f}")
    except Exception as e:
        print(f"Could not validate submission: {e}")
    
    return submission

if __name__ == "__main__":
    # Analyze the issue
    submissions, reference_ids = analyze_id_mismatch()
    
    # Check if analysis was successful
    if submissions is not None and reference_ids is not None:
        # Fix the submissions
        fix_submission_ids(submissions, reference_ids)
        
        # Create the ensemble with fixed submissions
        create_fixed_winning_ensemble()
    else:
        print("Analysis failed - cannot proceed with fixing submissions")
