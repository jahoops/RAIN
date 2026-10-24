import pandas as pd
import numpy as np
import os

# Load base models
best_rf = pd.read_csv('data/candidates_history/rf_boruta_winner_1741200309.csv')
next_best = pd.read_csv('data/candidates_history/submission_boruta_top_random_forest_score_0.882534.csv')

# Focus on micro-variations around 65/35
variations = {
    # Fine-tuning the ratio around the optimal 65/35
    'blend_66_34': best_rf['rainfall'] * 0.66 + next_best['rainfall'] * 0.34,
    'blend_64_36': best_rf['rainfall'] * 0.64 + next_best['rainfall'] * 0.36,
    'blend_63_37': best_rf['rainfall'] * 0.63 + next_best['rainfall'] * 0.37,
    
    # Ultra-nano scaling on the best 65/35
    'blend_65_35_ultra': (best_rf['rainfall'] * 0.65 + next_best['rainfall'] * 0.35) * 1.00025,
    
    # Create meta-blend of our best real-scoring submissions
    'meta_blend': best_rf['rainfall'] * 0.35 + next_best['rainfall'] * 0.30 + 
                 pd.read_csv('data/candidates/rf_blend_top.csv')['rainfall'] * 0.35,
}

# Save each variation
for name, predictions in variations.items():
    submission = pd.DataFrame({
        'id': best_rf['id'],
        'rainfall': predictions
    })
    
    submission_path = f'data/candidates/{name}_precision.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Saved {submission_path}")