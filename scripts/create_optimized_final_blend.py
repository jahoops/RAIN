import pandas as pd
import numpy as np
import os

# Load best tree model (most reliable across distributions)
best_tree = "/home/USER/Projects/RAIN/data/candidates/submission.85518.csv"
best_nn = "/home/USER/Projects/RAIN/data/candidates/submission_diverse_nn.csv"  # Best NN by reported score

tree_df = pd.read_csv(best_tree)
nn_df = pd.read_csv(best_nn)

# Create extremely conservative blend
# The tree model consistently outperforms on actual test data
tree_weight = 0.95
nn_weight = 0.05

# Create weighted blend
blended = pd.DataFrame({
    'id': tree_df.iloc[:, 0],
    'rainfall': tree_weight * tree_df.iloc[:, 1] + nn_weight * nn_df.iloc[:, 1]
})

# Save blend
output_path = '/home/USER/Projects/RAIN/data/candidates/final_distribution_robust_blend.csv'
blended.to_csv(output_path, index=False)
print(f"Created final blend with {tree_weight:.2f}/{nn_weight:.2f} tree/NN ratio")

# Also create a 97/3 blend for even more robustness
ultra_tree_weight = 0.97
ultra_nn_weight = 0.03

ultra_blended = pd.DataFrame({
    'id': tree_df.iloc[:, 0],
    'rainfall': ultra_tree_weight * tree_df.iloc[:, 1] + ultra_nn_weight * nn_df.iloc[:, 1]
})

ultra_path = '/home/USER/Projects/RAIN/data/candidates/ultra_robust_blend.csv'
ultra_blended.to_csv(ultra_path, index=False)
print(f"Created ultra-robust blend with {ultra_tree_weight:.2f}/{ultra_nn_weight:.2f} tree/NN ratio")