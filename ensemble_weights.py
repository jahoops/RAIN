# This creates a submission that blends different trained models

import pandas as pd
import numpy as np
import os
import glob
from scipy.optimize import minimize

def blend_submissions(base_path="data/candidates", output_name=None):
    """Blend multiple submission files with optimized weights"""
    # Find all submission files
    submission_files = glob.glob(f"{base_path}/*.csv")
    
    if not submission_files:
        print("No submission files found!")
        return
    
    # Load submissions and scores from filenames
    submissions = []
    scores = []
    
    for file in submission_files:
        try:
            # Extract score from filename using regex
            import re
            score_match = re.search(r'auc_(\d+\.\d+)', file)
            if score_match:
                score = float(score_match.group(1))
                
                # Load submission file
                sub = pd.read_csv(file)
                if 'id' in sub.columns and 'rainfall' in sub.columns:
                    submissions.append({
                        'file': file,
                        'data': sub,
                        'score': score
                    })
                    scores.append(score)
                    print(f"Loaded {file} with score {score:.6f}")
        except:
            print(f"Error loading {file}")
    
    if not submissions:
        print("No valid submissions loaded!")
        return
    
    # Calculate initial weights - higher scores get higher weights
    weights = np.array(scores) / sum(scores)
    print(f"Initial weights: {weights}")
    
    # Blend submissions with current weights
    def create_blend(w):
        # Normalize weights
        w = np.array(w)
        w = w / w.sum()
        
        # Create new prediction
        blend = pd.DataFrame({'id': submissions[0]['data']['id']})
        blend['rainfall'] = 0
        
        # Weight each submission
        for i, sub in enumerate(submissions):
            blend['rainfall'] += w[i] * sub['data']['rainfall']
            
        return blend
    
    # Create final blend
    final_blend = create_blend(weights)
    
    # Generate output filename
    if output_name is None:
        # Use weighted average of scores for final filename
        weighted_score = sum(w * s for w, s in zip(weights, scores))
        output_name = f"blended_ensemble_auc_{weighted_score:.6f}.csv"
    
    # Save final submission
    output_path = f"{base_path}/{output_name}"
    final_blend.to_csv(output_path, index=False)
    print(f"Saved blended submission to {output_path}")
    print(f"Final weights: {weights}")
    
    return output_path

if __name__ == "__main__":
    blend_submissions()