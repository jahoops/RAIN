import pandas as pd
import os
from data_pipeline import create_complete_data_pipeline

# Create the output directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Load data from your pipeline
print("Loading data from pipeline...")
_, X_test_temporal = create_complete_data_pipeline()

# Make sure we have the ID column
print(f"Test data has columns: {X_test_temporal.columns[:5]}")

# Save the test features with all columns
print(f"Saving test features with {X_test_temporal.shape[1]} columns...")
X_test_temporal.to_csv('data/test_features.csv', index=False)
print("Test features saved to data/test_features.csv")