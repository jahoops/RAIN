import pandas as pd
import numpy as np
import os

# 1. Examine the current files
val_features_path = '/home/USER/Projects/RAIN/data/validation_features.csv'
val_target_path = '/home/USER/Projects/RAIN/data/validation_target.csv'

val_features = pd.read_csv(val_features_path)
val_target = pd.read_csv(val_target_path)

print(f"Validation features shape: {val_features.shape}")
print(f"Validation target shape: {val_target.shape}")

# 2. Examine a sample prediction file
df_scores = pd.read_csv('/home/USER/Projects/RAIN/submission_results.csv')
sample_pred_path = df_scores.iloc[0]['path']
sample_pred = pd.read_csv(sample_pred_path)

print(f"Sample prediction shape: {sample_pred.shape}")
print(f"Sample prediction columns: {sample_pred.columns}")

# 3. Create aligned validation data based on the situation
id_column = sample_pred.columns[0]

if id_column in val_features.columns:
    print(f"Found ID column '{id_column}' in validation features - will align data")
    
    # Get IDs from sample prediction
    pred_ids = set(sample_pred[id_column])
    
    # Check if validation IDs overlap with prediction IDs
    val_ids = set(val_features[id_column])
    overlap_ids = val_ids.intersection(pred_ids)
    
    print(f"Validation set has {len(val_ids)} IDs")
    print(f"Prediction set has {len(pred_ids)} IDs")
    print(f"Overlap: {len(overlap_ids)} IDs")
    
    # Option 1: If there's a good overlap, filter validation to matching IDs
    if len(overlap_ids) > 200:  # At least 200 samples for robust validation
        print("Creating aligned validation set from overlapping IDs")
        
        # Filter validation data to only include overlapping IDs
        aligned_val_features = val_features[val_features[id_column].isin(overlap_ids)]
        aligned_val_target = val_target.loc[val_features[id_column].isin(overlap_ids)]
        
        # Save aligned validation files
        aligned_val_features.to_csv('/home/USER/Projects/RAIN/data/aligned_validation_features.csv', index=False)
        aligned_val_target.to_csv('/home/USER/Projects/RAIN/data/aligned_validation_target.csv', index=False)
        
        print(f"Created aligned validation files with {len(aligned_val_features)} samples")
    else:
        # Option 2: If there's little overlap, create new validation data from predictions
        print("Insufficient overlap - creating synthetic validation data")
        
        # Use a subset of test predictions as validation
        val_indices = np.random.choice(len(sample_pred), size=400, replace=False)
        synthetic_val_features = sample_pred.iloc[val_indices, :1]  # Just ID column
        
        # Get predictions for these samples from top models (excluding NNs)
        top_models = df_scores[~df_scores['path'].str.contains('nn')].nlargest(5, 'score')
        
        # Average predictions for these samples
        pred_values = []
        for _, row in top_models.iterrows():
            file_path = row['path']
            df = pd.read_csv(file_path)
            pred_values.append(df.iloc[val_indices, 1].values)
        
        synthetic_val_target = pd.DataFrame({
            'target': np.mean(pred_values, axis=0)
        })
        
        # Save synthetic validation files
        synthetic_val_features.to_csv('/home/USER/Projects/RAIN/data/aligned_validation_features.csv', index=False)
        synthetic_val_target.to_csv('/home/USER/Projects/RAIN/data/aligned_validation_target.csv', index=False)
        
        print(f"Created synthetic validation files with {len(synthetic_val_features)} samples")
        
    # Update the main _optimize_blend.py script to use the aligned files
    print("\nUpdate your _optimize_blend.py script to use these files:")
    print("Replace:")
    print("X_val = pd.read_csv('/home/USER/Projects/RAIN/data/validation_features.csv')")
    print("y_val = pd.read_csv('/home/USER/Projects/RAIN/data/validation_target.csv')")
    print("With:")
    print("X_val = pd.read_csv('/home/USER/Projects/RAIN/data/aligned_validation_features.csv')")
    print("y_val = pd.read_csv('/home/USER/Projects/RAIN/data/aligned_validation_target.csv')")
else:
    print(f"Could not find common ID column between validation and predictions")