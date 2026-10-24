import pandas as pd
import numpy as np

# Use best tree model with tiny contributions from diverse neural network models
tree_model = pd.read_csv("/home/USER/Projects/RAIN/data/candidates/submission.85518.csv")
nn_models = [
    "/home/USER/Projects/RAIN/data/candidates/submission_diverse_nn.csv",
    "/home/USER/Projects/RAIN/data/candidates/submission_regularized_nn.csv",
    "/home/USER/Projects/RAIN/data/candidates/submission_synthetic_nn.csv"
]

nn_dfs = [pd.read_csv(path) for path in nn_models]

# Combine 94% tree model with 2% each from different neural networks
blend = pd.DataFrame({
    'id': tree_model.iloc[:, 0],
    'rainfall': 0.94 * tree_model.iloc[:, 1] + 
               0.02 * nn_dfs[0].iloc[:, 1] + 
               0.02 * nn_dfs[1].iloc[:, 1] + 
               0.02 * nn_dfs[2].iloc[:, 1]
})

blend.to_csv("/home/USER/Projects/RAIN/data/candidates/robust_diverse_ensemble.csv", index=False)