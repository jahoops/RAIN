import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

# Load validation data
X_val = pd.read_csv('/home/USER/Projects/RAIN/data/aligned_validation_features.csv')
y_val = pd.read_csv('/home/USER/Projects/RAIN/data/aligned_validation_target.csv')

print(f"Validation set size: {len(X_val)} samples")

# Load predictions for validation samples
df_scores = pd.read_csv('/home/USER/Projects/RAIN/submission_results.csv')

# Compare individual model performance on validation set
val_ids = set(X_val[X_val.columns[0]])
models_mse = []

for _, row in df_scores.iterrows():
    file_path = row['path']
    model_name = os.path.basename(file_path)
    df = pd.read_csv(file_path)
    
    # Filter predictions to only include validation IDs
    id_col = df.columns[0]
    filtered_df = df[df[id_col].isin(val_ids)]
    filtered_df = filtered_df.sort_values(by=id_col)
    
    # Calculate MSE on validation set
    val_mse = mean_squared_error(y_val.iloc[:, 0].values, filtered_df.iloc[:, 1].values)
    models_mse.append((model_name, 1-val_mse, row['score']))

# Print models sorted by validation performance
models_mse.sort(key=lambda x: x[1], reverse=True)
print("\nModel performance (validation vs reported):")
print("Model, Validation Score, Reported Score")
for model, val_score, rep_score in models_mse:
    print(f"{model}: {val_score:.6f} vs {rep_score:.6f}")

# Plot comparison
plt.figure(figsize=(12, 8))
plt.scatter([x[2] for x in models_mse], [x[1] for x in models_mse])

for i, (model, val_score, rep_score) in enumerate(models_mse):
    plt.annotate(model.split('_')[0], (rep_score, val_score))

plt.xlabel('Reported Score')
plt.ylabel('Validation Score')
plt.title('Reported vs Validation Performance')
plt.grid(True)
plt.savefig('/home/USER/Projects/RAIN/data/validation_analysis.png')