import pandas as pd
import numpy as np

def fix_winning_blend_scale():
    """Fix the scale issue in the winning blend submission"""
    print("Fixing winning blend scale issue...")
    
    try:
        # Load the problematic submission
        submission_path = 'data/submission_winning_blend.csv'
        submission = pd.read_csv(submission_path)
        
        # Check the current range of values
        current_min = submission['rainfall'].min()
        current_max = submission['rainfall'].max()
        current_mean = submission['rainfall'].mean()
        
        print(f"Current rainfall range: {current_min:.6f} to {current_max:.6f}, mean: {current_mean:.6f}")
        
        # If values are outside the 0-1 range, normalize them
        if current_max > 1.5 or current_min < -0.5:
            print("Values out of expected range! Normalizing to 0-1 scale...")
            
            # Simple min-max normalization to 0-1 range
            submission['rainfall'] = (submission['rainfall'] - submission['rainfall'].min()) / \
                                    (submission['rainfall'].max() - submission['rainfall'].min())
            
            # Save backup of original
            original_backup_path = 'data/submission_winning_blend_original.csv'
            pd.read_csv(submission_path).to_csv(original_backup_path, index=False)
            print(f"Original submission backed up to {original_backup_path}")
            
            # Save normalized version
            submission.to_csv(submission_path, index=False)
            
            # Also save a copy with a different name
            fixed_path = 'data/submission_winning_blend_fixed.csv'
            submission.to_csv(fixed_path, index=False)
            
            # Copy to keep directory
            import shutil
            import os
            keep_path = os.path.join('data', 'keep', 'submission_winning_blend_fixed.csv')
            shutil.copy2(fixed_path, keep_path)
            
            print(f"Normalized submission saved to {submission_path}")
            print(f"New rainfall range: {submission['rainfall'].min():.6f} to {submission['rainfall'].max():.6f}")
            print(f"Fixed copy also saved to {fixed_path} and {keep_path}")
            
            # Validate the fixed submission
            try:
                from validate_submission import validate_submission
                print("\nValidating fixed submission...")
                result = validate_submission(fixed_path)
                if result and 'estimated_score' in result:
                    print(f"New estimated score: {result['estimated_score']:.6f}")
            except Exception as e:
                print(f"Error validating fixed submission: {str(e)}")
                
        else:
            print("Submission values appear to be in the expected range. No fix needed.")
            
    except Exception as e:
        print(f"Error fixing submission: {str(e)}")

if __name__ == "__main__":
    fix_winning_blend_scale()
