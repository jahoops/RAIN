import pandas as pd
import numpy as np

# Only use top tree-based models (exclude NN models)
top_trees = [
    "/home/USER/Projects/RAIN/data/candidates/submission.85518.csv",
    "/home/USER/Projects/RAIN/data/candidates/ensemble_rank_weighted.csv",
    "/home/USER/Projects/RAIN/data/candidates/submission_recommended_ensemble.csv",
    "/home/USER/Projects/RAIN/data/candidates/improved_synthetic_test.csv"
]

# Load and create weighted average (heavier weights to better models)
weights = [0.40, 0.30, 0.20, 0.10]  
dfs = [pd.read_csv(path) for path in top_trees]
template_df = dfs[0]

# Create weighted blend
ensemble = np.zeros(len(template_df))
for i, df in enumerate(dfs):
    ensemble += weights[i] * df.iloc[:, 1].values

# Save result
output = pd.DataFrame({template_df.columns[0]: template_df.iloc[:, 0], 
                      template_df.columns[1]: ensemble})
output.to_csv("/home/USER/Projects/RAIN/data/candidates/ultimate_tree_ensemble.csv", index=False)