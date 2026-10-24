import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import os
import time

def add_basic_seasonality(df):
    """Add only basic month/season features without complex transformations"""
    result = df.copy()
    
    # Try to extract or create month information
    if 'month' in result.columns:
        pass  # Use existing
    elif 'day' in result.columns and result['day'].max() > 31:
        result['month'] = ((result['day'] - 1) // 30) + 1
        result['month'] = result['month'].clip(1, 12)
    elif 'id' in result.columns:
        result['month'] = ((result['id'] % 365) // 30) + 1
        result['month'] = result['month'].clip(1, 12)
    
    # Create simple seasonal indicators if month exists
    if 'month' in result.columns:
        # Simple season indicator (1-4)
        result['season'] = ((result['month'] - 1) // 3) + 1
        
        # Add quarter indicators (more straightforward for trees)
        for q in range(1, 5):
            months_in_quarter = list(range((q-1)*3+1, (q-1)*3+4))
            result[f'quarter_{q}'] = result['month'].isin(months_in_quarter).astype(int)
    
    return result

def preserve_continuity(train_rainfall, pred_rainfall):
    """Apply minimal adjustments that preserve continuity of values"""
    # First check and fix any range issues
    pred_rainfall = scale_predictions(pred_rainfall)
    
    # Rest of your function remains the same...
    zero_pct = (train_rainfall == 0).mean()
    print(f"Zero percentage in training: {zero_pct*100:.2f}%")
    
    # Find threshold that would give approximately the same zero percentage
    threshold = max(0.0001, np.percentile(pred_rainfall, zero_pct * 92))
    
    # Apply a SOFT threshold (keeps very small values)
    calibrated = pred_rainfall.copy()
    calibrated[calibrated < threshold * 0.5] = 0
    
    # Never use exactly 1.0 (add tiny noise to ones)
    calibrated[calibrated > 0.999] = 0.999 + np.random.uniform(0, 0.001, 
                                                           size=np.sum(calibrated > 0.999))
    
    # Ensure high precision
    calibrated = np.round(calibrated, 16)
    
    return calibrated

def scale_predictions(predictions):
    """Scale predictions to [0,1] range while preserving distribution shape"""
    # Identify if we have a range problem
    min_val = predictions.min()
    max_val = predictions.max()
    
    if min_val < 0 or max_val > 1:
        print(f"Range issue detected! Raw predictions: [{min_val:.6f}, {max_val:.6f}]")
        
        # Apply proper min-max scaling instead of clipping
        scaled = (predictions - min_val) / (max_val - min_val)
        return scaled
    else:
        # No scaling needed
        return predictions

# Load data
print("Loading data...")
train = pd.read_csv('/home/USER/Projects/RAIN/data/train.csv')
test = pd.read_csv('/home/USER/Projects/RAIN/data/test.csv')

# Add basic seasonality
train = add_basic_seasonality(train)
test = add_basic_seasonality(test)

# Prepare data
y = train['rainfall']
X = train.drop(['id', 'rainfall'], axis=1)
X_test = test.drop(['id'], axis=1)

# Align columns
for col in set(X.columns) - set(X_test.columns):
    X_test[col] = 0
X_test = X_test[X.columns]

print(f"Training with {X.shape[1]} features")

# Train multiple models with increased depth but NO regularization
# to match the complex patterns in your top submission
print("\nTraining Random Forest...")
rf = RandomForestRegressor(
    n_estimators=1000, 
    max_depth=15,  # Deeper trees
    min_samples_leaf=2,  # More specific leaf nodes
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)
rf.fit(X, y)
rf_pred = rf.predict(X_test)

print("\nTraining XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=600,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=0.5,  # L2 regularization
    reg_alpha=0.1,   # L1 regularization
    random_state=43,
    objective='reg:squarederror'  # Explicitly use regression objective
)
xgb_model.fit(X, y)
xgb_pred = xgb_model.predict(X_test)

print("\nTraining LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=600,
    max_depth=10,  # Deeper
    learning_rate=0.03,
    subsample=0.9,  # Less regularization
    colsample_bytree=0.9,  # Less regularization
    random_state=44
)
lgb_model.fit(X, y)
lgb_pred = lgb_model.predict(X_test)

# Create ensemble with precise weights to match top submission pattern
print("\nCreating ensemble...")
ensemble_pred = 0.45 * rf_pred + 0.3 * xgb_pred + 0.25 * lgb_pred

# Add this after generating model predictions but before any calibration

# Check raw prediction ranges before any processing
print("\nRaw prediction ranges:")
print(f"RF predictions - Min: {rf_pred.min():.6f}, Max: {rf_pred.max():.6f}")
print(f"XGB predictions - Min: {xgb_pred.min():.6f}, Max: {xgb_pred.max():.6f}")
print(f"LGB predictions - Min: {lgb_pred.min():.6f}, Max: {lgb_pred.max():.6f}")
print(f"Ensemble predictions - Min: {ensemble_pred.min():.6f}, Max: {ensemble_pred.max():.6f}")

# If many predictions are outside [0,1], print the percentage
outside_range = ((ensemble_pred < 0) | (ensemble_pred > 1)).mean() * 100
print(f"Predictions outside [0,1] range: {outside_range:.2f}%")

# Apply minimal adjustments that preserve continuity
calibrated_pred = preserve_continuity(y, ensemble_pred)

# Scale predictions to [0,1] range
scaled_pred = scale_predictions(calibrated_pred)

# Generate submission
timestamp = int(time.time())
submission = pd.DataFrame({
    'id': test['id'],
    'rainfall': scaled_pred
})

# Save submission
os.makedirs('data/candidates', exist_ok=True)
submission_path = f'data/candidates/continuous_winner_{timestamp}.csv'
submission.to_csv(submission_path, index=False)

# Print matching statistics to your top submission
print("\nSubmission statistics:")
print(f"Min: {scaled_pred.min():.16f}")
print(f"Max: {scaled_pred.max():.16f}")
print(f"Mean: {scaled_pred.mean():.16f}")
print(f"Zero count: {(scaled_pred == 0).sum()} ({(scaled_pred == 0).mean()*100:.2f}%)")
print(f"Exactly 1.0 count: {(scaled_pred == 1.0).sum()}")

print(f"Values < 0.1: {(scaled_pred < 0.1).sum()} ({(scaled_pred < 0.1).mean()*100:.2f}%)")
print(f"Unique values: {len(np.unique(scaled_pred))}")
print(f"Submission saved to {submission_path}")