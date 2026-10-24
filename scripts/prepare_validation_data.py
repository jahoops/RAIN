import pandas as pd
import os

# Look for original dataset files
data_dir = "/home/USER/Projects/RAIN/data"

# Assuming you have train/test data like most ML projects
try:
    # Try to find training data with targets
    # Common filenames for training data
    possible_train_files = [
        "train.csv"    
    ]
    
    # Find a suitable training file
    train_file = None
    for filename in possible_train_files:
        full_path = os.path.join(data_dir, filename)
        if os.path.exists(full_path):
            train_file = full_path
            print(f"Found training data: {train_file}")
            break
    
    if not train_file:
        print("Could not find training data file.")
        exit(1)
    
    # Load the data
    train_data = pd.read_csv(train_file)
    
    # Identify target column (usually last column or named 'target', 'label', 'y', etc.)
    # This is an assumption - adjust based on your actual data
    target_col = train_data.columns[-1]  # Assume last column is target
    
    print(f"Using '{target_col}' as target column")
    
    # Split data into validation set (20% of training data)
    from sklearn.model_selection import train_test_split
    
    # Split off 20% as validation
    _, validation_data = train_test_split(
        train_data, test_size=0.2, random_state=42
    )
    
    # Create validation features and target
    X_val = validation_data.drop(columns=[target_col])
    y_val = validation_data[[target_col]]
    
    # Save validation files
    X_val.to_csv(os.path.join(data_dir, "validation_features.csv"), index=False)
    y_val.to_csv(os.path.join(data_dir, "validation_target.csv"), index=False)
    
    print(f"Created validation files with {len(X_val)} samples:")
    print(f"- {data_dir}/validation_features.csv")
    print(f"- {data_dir}/validation_target.csv")
    
except Exception as e:
    print(f"Error preparing validation data: {e}")