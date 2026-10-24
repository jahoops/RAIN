import pandas as pd
import numpy as np
import os
import glob
import shutil

def fix_final_submission():
    """Fix the final submission file with missing rainfall values"""
    print("=== FIXING FINAL SUBMISSION FILE ===")
    
    final_path = 'data/final/submission_final.csv'
    if not os.path.exists(final_path):
        print(f"Error: Final submission file not found at {final_path}")
        return False
    
    # Load the submission file
    final_df = pd.read_csv(final_path)
    print(f"Loaded submission from {final_path}")
    print(f"Shape: {final_df.shape}")
    
    # Check if rainfall column exists and has valid values
    if 'rainfall' not in final_df.columns:
        print("ERROR: 'rainfall' column is missing")
        has_valid_values = False
    else:
        missing_values = final_df['rainfall'].isna().sum()
        if missing_values > 0:
            print(f"WARNING: Found {missing_values} missing rainfall values")
            has_valid_values = False
        else:
            print("Rainfall values are complete")
            has_valid_values = True
    
    if not has_valid_values:
        print("\nAttempting to fix missing rainfall values...")
        
        # Strategy 1: Find the best alternative submission
        best_alternative = None
        best_score = 0
        
        # Check all submissions in keep directory
        keep_dir = 'data/keep'
        for submission_path in glob.glob(os.path.join(keep_dir, 'submission_*.csv')):
            try:
                df = pd.read_csv(submission_path)
                if 'id' in df.columns and 'rainfall' in df.columns:
                    if not df['rainfall'].isna().any():
                        # Check if this submission has values for all IDs in final_df
                        merged = final_df[['id']].merge(df, on='id', how='left')
                        missing = merged['rainfall'].isna().sum()
                        coverage = 1 - (missing / len(final_df))
                        
                        # Extract score if available in filename
                        filename = os.path.basename(submission_path)
                        score = 0
                        if 'score' in filename:
                            try:
                                score_str = filename.split('score_')[1].split('_')[0].split('.')[0]
                                score = float(score_str)
                            except:
                                pass
                        
                        print(f"Found alternative: {filename}, coverage: {coverage:.2%}, score: {score}")
                        
                        # Take this alternative if it has better coverage or better score with same coverage
                        if best_alternative is None or coverage > best_coverage or (coverage == best_coverage and score > best_score):
                            best_alternative = submission_path
                            best_coverage = coverage
                            best_score = score
            except Exception as e:
                print(f"Error checking {submission_path}: {str(e)}")
        
        if best_alternative:
            print(f"\nUsing best alternative: {os.path.basename(best_alternative)}")
            
            # Load the best alternative
            alt_df = pd.read_csv(best_alternative)
            
            # Create the fixed submission
            fixed_df = pd.DataFrame({'id': final_df['id']})
            fixed_df = fixed_df.merge(alt_df[['id', 'rainfall']], on='id', how='left')
            
            # Handle any remaining missing values
            if fixed_df['rainfall'].isna().any():
                print(f"Still have {fixed_df['rainfall'].isna().sum()} missing values after merge")
                print("Filling remaining missing values with 0.5")
                fixed_df['rainfall'] = fixed_df['rainfall'].fillna(0.5)
                
            # Save the fixed submission
            fixed_path = final_path.replace('.csv', '_fixed.csv')
            fixed_df.to_csv(fixed_path, index=False)
            print(f"Saved fixed submission to {fixed_path}")
            
            # Backup the original and replace it
            backup_path = final_path.replace('.csv', '_original_backup.csv')
            shutil.copy2(final_path, backup_path)
            print(f"Original file backed up to {backup_path}")
            
            fixed_df.to_csv(final_path, index=False)
            print(f"Fixed submission saved to {final_path}")
            
            return True
        else:
            print("\nCould not find a suitable alternative. Creating default submission...")
            
            # Create a default submission with all values set to 0.5
            fixed_df = pd.DataFrame({'id': final_df['id'], 'rainfall': 0.5})
            fixed_df.to_csv(final_path, index=False)
            print(f"Created default submission with all values set to 0.5")
            
            return True
    else:
        print("Final submission file looks good!")
        return True

def load_winning_model():
    """Try to find and load the best performing model"""
    # Look for models in various places
    candidates = [
        'data/submission_winning_ensemble_fixed3_fixed.csv',
        'data/submission_safe.csv',
        'data/submission_direct_ensemble.csv',
        'data/submission_ensemble.csv',
        'data/submission_synthetic_neural.csv',
        'data/submission_neural.csv',
        'data/submission_winning_blend.csv',
        'data/submission_suggested_ensemble.csv'
    ]
    
    # Also check keep directory
    keep_dir = 'data/keep'
    for file in os.listdir(keep_dir):
        if file.startswith('submission_') and file.endswith('.csv'):
            candidates.append(os.path.join(keep_dir, file))
    
    # Try each candidate
    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if 'id' in df.columns and 'rainfall' in df.columns and not df['rainfall'].isna().any():
                    print(f"Found working model: {path}")
                    return df
            except:
                pass
    
    # If nothing else works, create a default submission
    print("No working model found. Creating default submission.")
    
    # Load test IDs from test.csv
    try:
        test_df = pd.read_csv('data/test.csv')
        return pd.DataFrame({'id': test_df['id'], 'rainfall': 0.5})
    except:
        # If we can't even load test.csv, just return None
        return None

if __name__ == "__main__":
    print("Checking and fixing final submission file...")
    
    if not fix_final_submission():
        print("\nAutomatic fix failed. Attempting manual recovery...")
        
        winning_df = load_winning_model()
        if winning_df is not None:
            final_path = 'data/final/submission_final.csv'
            winning_df.to_csv(final_path, index=False)
            print(f"Recreated submission_final.csv using best available model")
            print(f"Saved to {final_path}")
        else:
            print("Could not load any working model. Please check your data files.")
