import pandas as pd
import numpy as np
import os

# Load base models
best_rf = pd.read_csv('data/candidates_history/rf_boruta_winner_1741200309.csv')
next_best = pd.read_csv('data/candidates_history/submission_boruta_top_random_forest_score_0.882534.csv')

# Validation-to-real gap compensation
# Since validation scores about 0.006 higher than real, 
# we need variations that score 0.862+ in validation

variations = {
    # Higher weight on best_rf (moving up from 65/35)
    'blend_67_33': best_rf['rainfall'] * 0.67 + next_best['rainfall'] * 0.33,
    'blend_70_30': best_rf['rainfall'] * 0.70 + next_best['rainfall'] * 0.30,
    
    # Microscopic scaling factors (much smaller than 1.001)
    'blend_65_35_micro': (best_rf['rainfall'] * 0.65 + next_best['rainfall'] * 0.35) * 1.0002,
    
    # Try loading a third model for triple blend
    # Recommended only if you have a third strong model
}

# Save each variation
os.makedirs('data/candidates', exist_ok=True)
for name, predictions in variations.items():
    submission = pd.DataFrame({
        'id': best_rf['id'],
        'rainfall': predictions
    })
    
    submission_path = f'data/candidates/{name}_final.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Saved {submission_path}")