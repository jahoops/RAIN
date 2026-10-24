import pandas as pd

# Even more conservative than ultra_robust
best_tree = "/home/USER/Projects/RAIN/data/candidates/submission.85518.csv"
best_nn = "/home/USER/Projects/RAIN/data/candidates/submission_diverse_nn.csv"

tree_df = pd.read_csv(best_tree)
nn_df = pd.read_csv(best_nn)

# 99/1 ratio - minimize neural network influence
blend = pd.DataFrame({
    'id': tree_df.iloc[:, 0],
    'rainfall': 0.99 * tree_df.iloc[:, 1] + 0.01 * nn_df.iloc[:, 1]
})

blend.to_csv("/home/USER/Projects/RAIN/data/candidates/super_conservative_blend.csv", index=False)