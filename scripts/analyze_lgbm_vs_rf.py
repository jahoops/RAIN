import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgbm
from data_pipeline import create_complete_data_pipeline

# Load data
X_train_temporal, _ = create_complete_data_pipeline()
y_train = X_train_temporal['rainfall']
X_features = X_train_temporal.drop(['id', 'rainfall'], axis=1)

# Train a simple RF and LGBM to compare feature importance
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_features, y_train)

lgb = lgbm.LGBMRegressor(n_estimators=100, random_state=42)
lgb.fit(X_features, y_train)

# Get feature importances
rf_importance = pd.DataFrame({
    'Feature': X_features.columns,
    'RF_Importance': rf.feature_importances_
}).sort_values('RF_Importance', ascending=False)

lgb_importance = pd.DataFrame({
    'Feature': X_features.columns,
    'LGBM_Importance': lgb.feature_importances_
}).sort_values('LGBM_Importance', ascending=False)

# Merge importance data
importance_df = pd.merge(rf_importance, lgb_importance, on='Feature')

# Calculate importance ratio (RF/LGBM)
importance_df['Importance_Ratio'] = importance_df['RF_Importance'] / importance_df['LGBM_Importance']
importance_df = importance_df.sort_values('Importance_Ratio', ascending=False)

# Print top differences
print("Features more important in RF than LGBM:")
print(importance_df.head(5))
print("\nFeatures more important in LGBM than RF:")
print(importance_df.tail(5))

# Create a blend leveraging both models
best_tree = pd.read_csv("/home/USER/Projects/RAIN/data/candidates/submission.85518.csv")
lgbm_preds = pd.read_csv("/home/USER/Projects/RAIN/data/candidates/optimized_lgbm_cv.csv")

# Create an asymmetric blend to leverage LGBM's different perspective
blend = pd.DataFrame({
    'id': best_tree.iloc[:, 0],
    'rainfall': 0.80 * best_tree.iloc[:, 1] + 0.20 * lgbm_preds.iloc[:, 1]
})

blend.to_csv("/home/USER/Projects/RAIN/data/candidates/rf_lgbm_blend.csv", index=False)

# Try a second blend with slightly different weighting
blend2 = pd.DataFrame({
    'id': best_tree.iloc[:, 0],
    'rainfall': 0.85 * best_tree.iloc[:, 1] + 0.15 * lgbm_preds.iloc[:, 1]
})

blend2.to_csv("/home/USER/Projects/RAIN/data/candidates/rf_lgbm_blend2.csv", index=False)