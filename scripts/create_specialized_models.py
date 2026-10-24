import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import lightgbm as lgbm
from data_pipeline import create_complete_data_pipeline

# Load data
X_train_temporal, X_test_temporal = create_complete_data_pipeline()
y_train = X_train_temporal['rainfall']

# Process features
X_train = X_train_temporal.drop(['id', 'rainfall'], axis=1)
X_test = X_test_temporal.drop(['id'], axis=1)

# Print available columns to avoid errors
print("\nAvailable features:")
print(X_train.columns.tolist())

# 1. Create a specialized RF model focused on short-term patterns
rf_cols = ['rainfall_rolling_3d_mean', 'cloud', 'rainfall_lag1', 'rainfall_lag2', 'humidity']
# Check if all columns exist
rf_cols = [col for col in rf_cols if col in X_train.columns]
print(f"\nUsing {len(rf_cols)} features for RF model: {rf_cols}")

rf_model = RandomForestRegressor(
    n_estimators=500, 
    max_features=0.8,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)
rf_model.fit(X_train[rf_cols], y_train)
rf_preds = rf_model.predict(X_test[rf_cols])

# 2. Create a specialized LightGBM model focused on meteorological factors
# Remove 'month' as it doesn't exist in the dataset
lgb_cols = ['winddirection', 'windspeed', 'temparature', 'day', 
           'rainfall_rolling_7d_mean', 'pressure']
# Check if all columns exist
lgb_cols = [col for col in lgb_cols if col in X_train.columns]
print(f"Using {len(lgb_cols)} features for LightGBM model: {lgb_cols}")

lgb_model = lgbm.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.01,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=43
)
lgb_model.fit(X_train[lgb_cols], y_train)
lgb_preds = lgb_model.predict(X_test[lgb_cols])

# 3. Create an ExtraTrees model using all features
et_model = ExtraTreesRegressor(
    n_estimators=500,
    max_features=0.7,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=44
)
et_model.fit(X_train, y_train)
et_preds = et_model.predict(X_test)

# Load best existing model
best_model = pd.read_csv("/home/USER/Projects/RAIN/data/candidates/submission.85518.csv")
best_preds = best_model.iloc[:, 1].values

# Create optimized blends leveraging the specialized models
# These weightings reflect the models' specialties
blend1 = pd.DataFrame({
    'id': X_test_temporal['id'],
    'rainfall': 0.70 * best_preds + 0.15 * rf_preds + 0.15 * lgb_preds
})
blend1.to_csv('/home/USER/Projects/RAIN/data/candidates/specialized_blend_1.csv', index=False)

# Try a more aggressive weighting of specialized models
blend2 = pd.DataFrame({
    'id': X_test_temporal['id'],
    'rainfall': 0.60 * best_preds + 0.20 * rf_preds + 0.15 * lgb_preds + 0.05 * et_preds
})
blend2.to_csv('/home/USER/Projects/RAIN/data/candidates/specialized_blend_2.csv', index=False)

# Create a more precise triple blend
blend3 = pd.DataFrame({
    'id': X_test_temporal['id'],
    'rainfall': 0.75 * best_preds + 0.10 * rf_preds + 0.15 * lgb_preds
})
blend3.to_csv('/home/USER/Projects/RAIN/data/candidates/specialized_blend_3.csv', index=False)

# Create a PREMIUM blend that might break 0.87
blend4 = pd.DataFrame({
    'id': X_test_temporal['id'],
    'rainfall': 0.78 * best_preds + 0.09 * rf_preds + 0.13 * lgb_preds
})
blend4.to_csv('/home/USER/Projects/RAIN/data/candidates/premium_blend.csv', index=False)

print("\nCreated specialized model blends leveraging different feature importances")