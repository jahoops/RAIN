# filepath: /home/USER/Projects/RAIN/winner_replication.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
import time
from data_loader import load_data, preprocess_data
from feature_engineering import engineer_features

# Load and preprocess data
print("Loading data...")
train_df, test_df = load_data()
train_df, test_df = preprocess_data(train_df, test_df)

# Apply the boruta_top feature engineering (one of your top strategies)
print("Applying feature engineering...")
train, test = engineer_features(train_df, test_df, strategy='boruta_top')

# Prepare data
X = train.drop(['id', 'rainfall'], axis=1, errors='ignore')
y = train['rainfall']
X_test = test.drop(['id'], axis=1, errors='ignore')

# Use the exact Random Forest configuration from your winning submissions
print("Training Random Forest model...")
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    min_samples_split=5,
    n_jobs=-1,
    random_state=42
)
rf.fit(X, y)
test_preds = rf.predict(X_test)

# Create submission
timestamp = int(time.time())
submission = pd.DataFrame({
    'id': test['id'],
    'rainfall': test_preds
})

# Save submission
submission_path = f'data/candidates/rf_boruta_winner_{timestamp}.csv'
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

# Also create the 'pure_winner' variant (slight scaling)
submission_pure = submission.copy()
submission_pure['rainfall'] *= 1.001  # Tiny scaling for 'pure_winner'
pure_submission_path = f'data/candidates/pure_rf_boruta_{timestamp}.csv'
submission_pure.to_csv(pure_submission_path, index=False)
print(f"Pure winner variant saved to {pure_submission_path}")