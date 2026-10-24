import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgbm
from data_pipeline import create_complete_data_pipeline

print("Loading data...")
X_train_temporal, X_test_temporal = create_complete_data_pipeline()
y_train = X_train_temporal['rainfall']

# Process features
X_train = X_train_temporal.drop(['id', 'rainfall'], axis=1)
X_test = X_test_temporal.drop(['id'], axis=1)

# Print available columns
print(f"\nAvailable features: {X_train.columns.tolist()}")

# Build a premium Random Forest with optimal parameters
print("\n1. Training premium Random Forest...")
rf_premium = RandomForestRegressor(
    n_estimators=1000,  # More trees
    max_depth=None,     # Unlimited depth 
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=0.7,
    bootstrap=True,
    oob_score=True,     # Use OOB for validation
    n_jobs=-1,
    random_state=87     # Special seed for potential breakthrough
)
rf_premium.fit(X_train, y_train)
rf_preds = rf_premium.predict(X_test)
print(f"RF OOB Score: {rf_premium.oob_score_:.6f}")

# Build a separate gradient boosting model 
print("\n2. Training GradientBoosting model...")
gbm = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    random_state=88
)
gbm.fit(X_train, y_train)
gbm_preds = gbm.predict(X_test)

# Load best existing model
print("\n3. Loading best existing model...")
best_model = pd.read_csv("/home/USER/Projects/RAIN/data/candidates/submission.85518.csv")
best_preds = best_model.iloc[:, 1].values

# Create an optimal blend for potential 0.87 breakthrough
breakthrough_blend = pd.DataFrame({
    'id': X_test_temporal['id'],
    'rainfall': 0.8 * best_preds + 0.1 * rf_preds + 0.1 * gbm_preds
})
breakthrough_blend.to_csv('/home/USER/Projects/RAIN/data/candidates/breakthrough_blend_direct.csv', index=False)
print("Created BREAKTHROUGH_BLEND_DIRECT")

# Create a more aggressive variant 
aggressive_blend = pd.DataFrame({
    'id': X_test_temporal['id'],
    'rainfall': 0.75 * best_preds + 0.15 * rf_preds + 0.10 * gbm_preds
})
aggressive_blend.to_csv('/home/USER/Projects/RAIN/data/candidates/breakthrough_aggressive.csv', index=False)
print("Created BREAKTHROUGH_AGGRESSIVE")

print("\nCompleted creation of direct breakthrough approaches")