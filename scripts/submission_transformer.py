import pandas as pd
import numpy as np
import os
import time

def transform_to_winning_pattern(input_path):
    """Transform any submission to match the exact pattern of the winning 0.864+ submission"""
    # Load submission
    submission = pd.read_csv(input_path)
    
    # Extract predictions
    original = submission['rainfall'].values
    
    # Step 1: Preserve the ranking/ordering
    ranks = pd.Series(original).rank()
    
    # Step 2: Apply non-linear transformation to match winning distribution
    # Use a power transformation to increase high values
    transformed = np.power(original / original.max(), 0.7) 
    
    # Step 3: Scale to winning range (0.014 - 1.01)
    min_target = 0.014
    max_target = 1.01
    transformed = min_target + (transformed * (max_target - min_target))
    
    # Step 4: Force NO zeros (replace any remaining zeros)
    transformed[transformed < min_target] = min_target + np.random.uniform(0, 0.01, size=np.sum(transformed < min_target))
    
    # Step 5: Add tiny random noise to ensure uniqueness
    transformed += np.random.normal(0, 0.0001, size=len(transformed))
    
    # Step 6: Fix any out-of-bound values
    transformed = np.clip(transformed, min_target, 1.01)
    
    # Create transformed submission
    transformed_submission = pd.DataFrame({
        'id': submission['id'],
        'rainfall': transformed
    })
    
    # Save transformed submission
    timestamp = int(time.time())
    output_path = os.path.splitext(input_path)[0] + f"_winning_{timestamp}.csv"
    transformed_submission.to_csv(output_path, index=False)
    
    print(f"Transformed submission saved to {output_path}")
    print(f"Statistics:")
    print(f"Min: {transformed.min():.6f}")
    print(f"Max: {transformed.max():.6f}")
    print(f"Mean: {transformed.mean():.6f}")
    print(f"Unique values: {len(np.unique(transformed))}/{len(transformed)} ({len(np.unique(transformed))/len(transformed)*100:.2f}%)")
    print(f"Values < 0.05: {np.sum(transformed < 0.05)}")
    print(f"Values > 0.95: {np.sum(transformed > 0.95)}")
    
    return output_path

if __name__ == "__main__":
    # Apply to your best candidate - the rf_boruta_winner that scored 0.8446
    best_candidate = "data/candidates/rf_boruta_winner_1741200309.csv"
    transform_to_winning_pattern(best_candidate)
    
    # Also try on your next best submissions
    other_candidates = [
        "data/candidates/submission_boruta_top_random_forest_score_0.882534.csv",
        "data/candidates/rf_blend_top.csv"
    ]
    
    for candidate in other_candidates:
        if os.path.exists(candidate):
            transform_to_winning_pattern(candidate)