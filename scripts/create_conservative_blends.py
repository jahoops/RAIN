import pandas as pd
import numpy as np
import os

# Load best tree model and best neural network model
best_tree = "/home/USER/Projects/RAIN/data/candidates/submission.85518.csv"
best_nn = "/home/USER/Projects/RAIN/data/candidates/submission_regularized_nn.csv"

# Verify if the new robust NN exists
robust_nn = "/home/USER/Projects/RAIN/data/candidates/submission_robust_nn.csv"
if os.path.exists(robust_nn):
    best_nn = robust_nn
    print("Using new robust neural network for blending")
else:
    print("Using previously regularized neural network for blending")

tree_df = pd.read_csv(best_tree)
nn_df = pd.read_csv(best_nn)

# Create multiple blends with different weightings
for tree_weight in [0.9, 0.85, 0.8]:
    nn_weight = 1 - tree_weight
    
    # Create weighted blend
    blended = pd.DataFrame({
        'id': tree_df.iloc[:, 0],
        'rainfall': tree_weight * tree_df.iloc[:, 1] + nn_weight * nn_df.iloc[:, 1]
    })
    
    # Save blend
    output_path = f'/home/USER/Projects/RAIN/data/candidates/tree{int(tree_weight*100)}_nn{int(nn_weight*100)}_blend.csv'
    blended.to_csv(output_path, index=False)
    print(f"Created {tree_weight:.2f}/{nn_weight:.2f} tree/NN blend: {output_path}")