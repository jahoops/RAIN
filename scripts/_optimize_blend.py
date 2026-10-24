import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os
import re
from scipy.optimize import minimize

# Load validation data
X_val = pd.read_csv('/home/USER/Projects/RAIN/data/aligned_validation_features.csv')
y_val = pd.read_csv('/home/USER/Projects/RAIN/data/aligned_validation_target.csv')

print(f"Validation data shapes: X_val = {X_val.shape}, y_val = {y_val.shape}")

# Get all paths and scores 
df_scores = pd.read_csv('/home/USER/Projects/RAIN/submission_results.csv')
scored_files = df_scores.sort_values(by='score', ascending=False)

# Filter out very low-scoring models to prevent them from dominating
min_score_threshold = 0.80
scored_files = scored_files[scored_files['score'] > min_score_threshold]

print(f"Using {len(scored_files)} models with scores > {min_score_threshold}")
for _, row in scored_files.iterrows():
    print(f"- {row['path']} (score: {row['score']:.4f})")

candidates_dir = '/home/USER/Projects/RAIN/data/candidates'

# Filter validation IDs
val_ids = set(X_val[X_val.columns[0]])
predictions_val = []   # For validation
predictions_full = []  # For final submission

for _, row in scored_files.iterrows():
    file_path = row['path']
    df = pd.read_csv(file_path)
    
    # Store full predictions for submission blend
    predictions_full.append(df.iloc[:, 1].values)
    
    # Filter predictions to only include validation IDs
    id_col = df.columns[0]
    filtered_df = df[df[id_col].isin(val_ids)]
    filtered_df = filtered_df.sort_values(by=id_col)
    
    predictions_val.append(filtered_df.iloc[:, 1].values)
    
    # Diagnostic info
    print(f"  Full predictions: {len(df)}, Validation predictions: {len(filtered_df)}")

predictions_val = np.array(predictions_val).T  # Shape: (n_val_samples, n_models)

# Print shapes for debugging
print(f"\nValidation array shape: {predictions_val.shape}")
print(f"Target array shape: {y_val.shape}")

# Ensure targets are aligned
if len(y_val) != predictions_val.shape[0]:
    print("WARNING: Validation target and predictions have different lengths!")

# Define objective function (MSE)
def objective(weights):
    weights = weights / np.sum(weights)
    blend = np.sum(predictions_val * weights, axis=1)
    return mean_squared_error(y_val.iloc[:, 0].values, blend)

# Initial weights (equal)
initial_weights = np.ones(len(predictions_val[0])) / len(predictions_val[0])

# Optimize weights
bounds = [(0, 1) for _ in range(len(initial_weights))]
result = minimize(objective, initial_weights, bounds=bounds, method='L-BFGS-B')

# Normalize final weights to sum to 1
optimal_weights = result.x / np.sum(result.x)
print("\nOptimal weights for blending:")
for i, ((_, row), weight) in enumerate(zip(scored_files.iterrows(), optimal_weights)):
    print(f"{row['path']}: {weight:.4f}")

# Create final optimized blend
blend = pd.read_csv(scored_files.iloc[0]['path'])  # Use first file as template
blend.iloc[:, 1] = np.sum(np.array(predictions_full).T * optimal_weights, axis=1)
blend.to_csv(os.path.join(candidates_dir, "another_optimized_blend.csv"), index=False)
print("\nCreated optimized blend: another_optimized_blend.csv")

# Calculate and print expected score
expected_mse = objective(optimal_weights)
print(f"Expected MSE on validation data: {expected_mse:.6f}")
print(f"Expected score: {1 - expected_mse:.6f}")