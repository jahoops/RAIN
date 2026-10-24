import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import lightgbm as lgbm
from data_pipeline import create_complete_data_pipeline

print("Loading data...")
X_train_temporal, X_test_temporal = create_complete_data_pipeline()
y_train = X_train_temporal['rainfall']

# Process features
X_train = X_train_temporal.drop(['id', 'rainfall'], axis=1)
X_test = X_test_temporal.drop(['id'], axis=1)

# Print available columns
print("\nAvailable features:")
print(X_train.columns.tolist())

# Identify feature groups based on importance analysis - but check existence first
# Define desired features
desired_rainfall_features = ['rainfall_rolling_3d_mean', 'rainfall_lag1', 'rainfall_lag2', 
                          'rainfall_lead1', 'rainfall_rolling_7d_mean']
desired_weather_features = ['cloud', 'humidity', 'pressure', 'temparature']
desired_wind_features = ['windspeed', 'winddirection']
desired_time_features = ['day']  # Remove 'month' as it doesn't exist

# Filter to only include features that exist in the dataset
rainfall_features = [col for col in desired_rainfall_features if col in X_train.columns]
weather_features = [col for col in desired_weather_features if col in X_train.columns]
wind_features = [col for col in desired_wind_features if col in X_train.columns]
time_features = [col for col in desired_time_features if col in X_train.columns]

print(f"\nUsing rainfall features: {rainfall_features}")
print(f"Using weather features: {weather_features}")
print(f"Using wind features: {wind_features}")
print(f"Using time features: {time_features}")

# Build specialized Random Forest for rainfall patterns
print("\n1. Training specialized RandomForest on rainfall patterns...")
rf_rainfall = RandomForestRegressor(
    n_estimators=400, 
    max_depth=None,
    min_samples_leaf=1,
    max_features=0.7,
    n_jobs=-1,
    random_state=42
)
rf_rainfall.fit(X_train[rainfall_features], y_train)
rf_rainfall_preds = rf_rainfall.predict(X_test[rainfall_features])

# Build specialized LightGBM for meteorological factors
print("\n2. Training specialized LightGBM on meteorological factors...")
lgb_weather = lgbm.LGBMRegressor(
    n_estimators=400,
    learning_rate=0.01,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=43
)
weather_wind_features = weather_features + wind_features + time_features
print(f"Using {len(weather_wind_features)} features for LightGBM: {weather_wind_features}")
lgb_weather.fit(X_train[weather_wind_features], y_train)
lgb_weather_preds = lgb_weather.predict(X_test[weather_wind_features])

# Build ExtraTrees on all features for a different perspective
print("\n3. Training ExtraTrees on all features...")
et_all = ExtraTreesRegressor(
    n_estimators=400,
    max_features=0.7,
    min_samples_leaf=1,
    bootstrap=True,
    n_jobs=-1,
    random_state=44
)
et_all.fit(X_train, y_train)
et_all_preds = et_all.predict(X_test)

# Load best existing model
print("\n4. Loading best existing model and creating premium blends...")
best_model = pd.read_csv("/home/USER/Projects/RAIN/data/candidates/submission.85518.csv")
best_preds = best_model.iloc[:, 1].values

# Try several different weights to find the optimal blend

# Create SPECIALIZED_PREMIUM_BLEND - weighted carefully to focus on strengths
premium_blend = pd.DataFrame({
    'id': X_test_temporal['id'],
    'rainfall': 0.76 * best_preds + 0.08 * rf_rainfall_preds + 
                0.10 * lgb_weather_preds + 0.06 * et_all_preds
})
premium_blend.to_csv('/home/USER/Projects/RAIN/data/candidates/specialized_premium_blend.csv', index=False)
print("Created SPECIALIZED_PREMIUM_BLEND")

# Create HYPER_SPECIALIZED_BLEND - more aggressive use of specialized models
hyper_blend = pd.DataFrame({
    'id': X_test_temporal['id'],
    'rainfall': 0.70 * best_preds + 0.12 * rf_rainfall_preds + 
                0.13 * lgb_weather_preds + 0.05 * et_all_preds
})
hyper_blend.to_csv('/home/USER/Projects/RAIN/data/candidates/hyper_specialized_blend.csv', index=False)
print("Created HYPER_SPECIALIZED_BLEND")

# Create a CONSERVATIVE_SPECIALIZED_BLEND - higher weight on best model
conservative_blend = pd.DataFrame({
    'id': X_test_temporal['id'],
    'rainfall': 0.82 * best_preds + 0.06 * rf_rainfall_preds + 
                0.08 * lgb_weather_preds + 0.04 * et_all_preds
})
conservative_blend.to_csv('/home/USER/Projects/RAIN/data/candidates/conservative_specialized_blend.csv', index=False)
print("Created CONSERVATIVE_SPECIALIZED_BLEND")

# Create a RAINFALL_FOCUSED_BLEND - emphasizing rainfall patterns
rainfall_focused_blend = pd.DataFrame({
    'id': X_test_temporal['id'],
    'rainfall': 0.75 * best_preds + 0.15 * rf_rainfall_preds + 
                0.07 * lgb_weather_preds + 0.03 * et_all_preds
})
rainfall_focused_blend.to_csv('/home/USER/Projects/RAIN/data/candidates/rainfall_focused_blend.csv', index=False)
print("Created RAINFALL_FOCUSED_BLEND")

print("\nCompleted creation of specialized model blends")