import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# 1. Create validation data from original training set (if available)
try:
    # Try to find original training data
    data_dir = "/home/USER/Projects/RAIN/data"
    training_files = ["train.csv", "training.csv", "train_data.csv"]
    
    train_file = None
    for file in training_files:
        if os.path.exists(os.path.join(data_dir, file)):
            train_file = os.path.join(data_dir, file)
            break
            
    if train_file:
        print(f"Found training file: {train_file}")
        train_data = pd.read_csv(train_file)
        
        # Split into training and validation
        train_part, val_part = train_test_split(train_data, test_size=0.2, random_state=42)
        
        # Separate features and target in validation set
        # Assuming last column is target and first is ID
        val_features = val_part.iloc[:, :-1]
        val_target = val_part.iloc[:, -1:] 
        
        # Save proper validation files
        val_features.to_csv(os.path.join(data_dir, "validation_features.csv"), index=False)
        val_target.to_csv(os.path.join(data_dir, "validation_target.csv"), index=False)
        
        print("Created validation files from original training data")
    else:
        # 2. Alternative: Create validation from consensus of top models
        print("No training file found. Creating validation from model consensus...")
        
        # Get top submissions (excluding neural networks)
        df_scores = pd.read_csv('/home/USER/Projects/RAIN/submission_results.csv')
        top_models = df_scores[~df_scores['path'].str.contains('nn')].nlargest(5, 'score')
        
        predictions = []
        for _, row in top_models.iterrows():
            file_path = row['path']
            df = pd.read_csv(file_path)
            predictions.append(df.iloc[:, 1].values)
        
        # Create pseudo-target as average of top models
        consensus = np.mean(predictions, axis=0)
        
        # Create validation features (from test file metadata)
        test_file = os.path.join(data_dir, "test.csv")
        if os.path.exists(test_file):
            test_data = pd.read_csv(test_file)
            
            # Create validation files
            test_data.to_csv(os.path.join(data_dir, "validation_features.csv"), index=False)
            pd.DataFrame({'target': consensus}).to_csv(os.path.join(data_dir, "validation_target.csv"), index=False)
            
            print("Created validation files from model consensus")
        else:
            print("ERROR: Could not find test.csv or training data")
            
except Exception as e:
    print(f"Error creating validation data: {e}")