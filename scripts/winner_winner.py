# filepath: /home/USER/Projects/RAIN/pure_rf_top.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Use exact same data preparation as in run_experiments.py
# that created submission_boruta_top_random_forest_score_0.882534.csv

# Simple blend formula from your top performers
best_rf = pd.read_csv('data/candidates_history/rf_boruta_winner_1741200309.csv')
next_best = pd.read_csv('data/candidates_history/submission_boruta_top_random_forest_score_0.882534.csv')

blend = pd.DataFrame({
    'id': best_rf['id'],
    'rainfall': best_rf['rainfall'] * 0.65 + next_best['rainfall'] * 0.35
})

blend.to_csv('data/candidates/rf_blend_top.csv', index=False)

# More blend variations to try
blend_weights = [
    (0.70, 0.30),  # Slightly more weight on best_rf
    (0.60, 0.40),  # Slightly less weight on best_rf
    (0.75, 0.25)   # Even more weight on best_rf
]

for i, (w1, w2) in enumerate(blend_weights):
    blend = pd.DataFrame({
        'id': best_rf['id'],
        'rainfall': best_rf['rainfall'] * w1 + next_best['rainfall'] * w2
    })
    blend.to_csv(f'data/candidates/rf_blend_variant_{i}.csv', index=False)

# Triple blend with the third RF model
rf_minimal = pd.read_csv('data/candidates_history/submission_minimal_random_forest_score_0.882534.csv')

triple_blend = pd.DataFrame({
    'id': best_rf['id'],
    'rainfall': (best_rf['rainfall'] * 0.60 + 
                next_best['rainfall'] * 0.25 + 
                rf_minimal['rainfall'] * 0.15)
})
triple_blend.to_csv('data/candidates/rf_triple_blend.csv', index=False)

# RF parameters that matched your top models
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=4,
    min_samples_split=5,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)

# The pure_winner used 0.1% scaling
scaled = pd.DataFrame({
    'id': best_rf['id'],
    'rainfall': best_rf['rainfall'] * 1.001
})

scaled.to_csv('data/candidates/rf_scaled_top.csv', index=False)