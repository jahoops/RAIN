import pandas as pd

# Try a super-aggressive blend skewed heavily toward the best performer
best_model = pd.read_csv("/home/USER/Projects/RAIN/data/candidates/submission.85518.csv")
best_ensemble = pd.read_csv("/home/USER/Projects/RAIN/data/candidates/ensemble_rank_weighted.csv")

adjusted = pd.DataFrame({
    'id': best_model.iloc[:, 0],
    'rainfall': 0.95 * best_model.iloc[:, 1] + 0.05 * best_ensemble.iloc[:, 1]
})

adjusted.to_csv('/home/USER/Projects/RAIN/data/candidates/adjusted_winner.csv', index=False)
print("Created adjusted winner submission")