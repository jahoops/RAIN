import pandas as pd
import numpy as np
import os

# Load your base models
best_rf = pd.read_csv('data/candidates_history/rf_boruta_winner_1741200309.csv')
next_best = pd.read_csv('data/candidates_history/submission_boruta_top_random_forest_score_0.882534.csv')

# Try several targeted variations to find the 0.856+ sweet spot
variations = {
    # Shift weights slightly more toward next_best (which had 0.8547 validation)
    'blend_60_40': best_rf['rainfall'] * 0.60 + next_best['rainfall'] * 0.40,
    'blend_55_45': best_rf['rainfall'] * 0.55 + next_best['rainfall'] * 0.45,
    
    # Apply tiny scaling factors to your blend
    'blend_65_35_scaled': (best_rf['rainfall'] * 0.65 + next_best['rainfall'] * 0.35) * 1.001,
    
    # Try triple blends with a third model
    # Load the third model - either minimal or all feature set
    # 'triple_blend': best_rf['rainfall'] * 0.55 + next_best['rainfall'] * 0.30 + 
    #                third_model['rainfall'] * 0.15
}

# Save each variation
os.makedirs('data/candidates', exist_ok=True)
for name, predictions in variations.items():
    submission = pd.DataFrame({
        'id': best_rf['id'],
        'rainfall': predictions
    })
    
    submission_path = f'data/candidates/{name}_breakthrough.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Saved {submission_path}")