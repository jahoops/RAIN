import pandas as pd
import numpy as np

# This approach explicitly adjusts for the observed distribution shift
best_tree = pd.read_csv("/home/USER/Projects/RAIN/data/candidates/submission.85518.csv")
nn_model = pd.read_csv("/home/USER/Projects/RAIN/data/candidates/submission_diverse_nn.csv")

# Calculate mean and std of each model's predictions
tree_mean = np.mean(best_tree.iloc[:, 1])
tree_std = np.std(best_tree.iloc[:, 1])
nn_mean = np.mean(nn_model.iloc[:, 1])
nn_std = np.std(nn_model.iloc[:, 1])

# Adjust neural network predictions to match tree distribution
nn_adjusted = (nn_model.iloc[:, 1] - nn_mean) * (tree_std / nn_std) + tree_mean

# Create a blend with distribution-adjusted NN predictions
blend = pd.DataFrame({
    'id': best_tree.iloc[:, 0],
    'rainfall': 0.92 * best_tree.iloc[:, 1] + 0.08 * nn_adjusted
})

blend.to_csv("/home/USER/Projects/RAIN/data/candidates/distribution_adjusted_blend.csv", index=False)