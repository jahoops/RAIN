import pandas as pd
import numpy as np
import os

# Find test data used for submissions
data_dir = "/home/USER/Projects/RAIN/data"
candidates_dir = "/home/USER/Projects/RAIN/data/candidates"

# Get top submission files 
candidate_files = [f for f in os.listdir(candidates_dir) if f.endswith('.csv')][:5]

# Load test data (features)
# Try common test file names
test_files = ["test.csv"]
test_file = None

for filename in test_files:
    full_path = os.path.join(data_dir, filename)
    if os.path.exists(full_path):
        test_file = full_path
        print(f"Found test data: {test_file}")
        break

if not test_file:
    print("Could not find test data file. Creating synthetic validation data instead.")
    # Create synthetic validation data
    submission = pd.read_csv(os.path.join(candidates_dir, candidate_files[0]))
    id_col = submission.columns[0]
    
    # Create features (just IDs for now as placeholder)
    X_val = pd.DataFrame({id_col: submission[id_col]})
    
    # Create pseudo-targets as average of top submissions
    preds = []
    for file in candidate_files:
        df = pd.read_csv(os.path.join(candidates_dir, file))
        preds.append(df.iloc[:, 1].values)
    
    # Use average as pseudo-ground-truth
    y_val = pd.DataFrame({
        'target': np.mean(preds, axis=0)
    })
else:
    # Use test data as validation features
    X_val = pd.read_csv(test_file)
    
    # Create pseudo-targets as average of top submissions
    preds = []
    id_col = None
    for file in candidate_files:
        df = pd.read_csv(os.path.join(candidates_dir, file))
        if id_col is None:
            id_col = df.columns[0]
        preds.append(df.iloc[:, 1].values)
    
    # Use average as pseudo-ground-truth
    y_val = pd.DataFrame({
        'target': np.mean(preds, axis=0)
    })

# Save validation files
X_val.to_csv(os.path.join(data_dir, "validation_features.csv"), index=False)
y_val.to_csv(os.path.join(data_dir, "validation_target.csv"), index=False)

print(f"Created validation files with {len(X_val)} samples:")
print(f"- {data_dir}/validation_features.csv")
print(f"- {data_dir}/validation_target.csv")
print("\nNOTE: Using pseudo-labels derived from model predictions.")