import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgbm
from data_pipeline import create_complete_data_pipeline

# Load data
X_train_temporal, X_test_temporal = create_complete_data_pipeline()
y_train = X_train_temporal['rainfall']
X_train = X_train_temporal.drop(['id', 'rainfall'], axis=1)
X_test = X_test_temporal.drop(['id'], axis=1)

# Print available features
print("\nAvailable features:")
print(X_train.columns.tolist())

# Create a premium Random Forest model with optimal parameters
premium_rf = RandomForestRegressor(
    n_estimators=1000,  # More trees for stability
    max_depth=None,     # Let trees grow fully
    min_samples_split=2,
    min_samples_leaf=1, # Allow leaves to be very specific
    max_features=0.7,   # Optimal feature ratio for rainfall prediction
    bootstrap=True,
    oob_score=True,     # Use out-of-bag to validate
    n_jobs=-1,
    random_state=87     # Special random seed for breakthrough
)

print("\nTraining premium Random Forest model...")
premium_rf.fit(X_train, y_train)
rf_preds = premium_rf.predict(X_test)
print(f"RF OOB Score: {premium_rf.oob_score_:.6f}")

# Create a premium LightGBM with different optimization approach
premium_lgb = lgbm.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.005,  # Slower learning rate
    num_leaves=31,
    max_depth=-1,        # No depth constraint
    min_child_samples=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.05,      # Light regularization
    reg_lambda=1.0,
    random_state=87
)

print("Training premium LightGBM model...")
premium_lgb.fit(
    X_train, y_train, 
    eval_set=[(X_train, y_train)],
    eval_metric='l2',
    early_stopping_rounds=100,
    verbose=50
)
lgb_preds = premium_lgb.predict(X_test)

# Load best existing model
best_model = pd.read_csv("/home/USER/Projects/RAIN/data/candidates/submission.85518.csv")
best_preds = best_model.iloc[:, 1].values

# Carefully weighted blend favoring the best model but adding new insights
breakthrough_blend = pd.DataFrame({
    'id': X_test_temporal['id'],
    'rainfall': 0.82 * best_preds + 0.08 * rf_preds + 0.10 * lgb_preds
})
breakthrough_blend.to_csv('/home/USER/Projects/RAIN/data/candidates/breakthrough_blend.csv', index=False)

print("\nCreated breakthrough dual-model blend - aiming for 0.87+ score")