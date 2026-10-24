import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import os
import time

def add_basic_seasonality(df):
    """Add only basic month/season features without any complex transformations"""
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

def maintain_natural_distribution(predictions):
    """Keep predictions continuous while enforcing zero pattern"""
    # Get zero percentage from training data
    train = pd.read_csv('/home/USER/Projects/RAIN/data/train.csv')
    zero_pct = (train['rainfall'] == 0).mean()
    
    # Sort predictions to find threshold
    threshold = np.percentile(predictions, zero_pct * 100)
    
    # Apply zero threshold
    calibrated = predictions.copy()
    calibrated[calibrated < threshold] = 0
    
    # Important: Do NOT binarize the remaining values!
    # Keep them as continuous predictions
    
    return calibrated

# Load data
print("Loading data...")
train = pd.read_csv('/home/USER/Projects/RAIN/data/train.csv')
test = pd.read_csv('/home/USER/Projects/RAIN/data/test.csv')

# Get zero percentage
zero_pct = (train['rainfall'] == 0).mean()
print(f"Zero percentage in training: {zero_pct*100:.2f}%")

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

# Train models without complex stacking
print("\nTraining Random Forest...")
rf = RandomForestRegressor(
    n_estimators=1000, 
    max_depth=12,
    min_samples_leaf=5,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)
rf.fit(X, y)
rf_pred = rf.predict(X_test)

print("\nTraining XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=43
)
xgb_model.fit(X, y)
xgb_pred = xgb_model.predict(X_test)

print("\nTraining LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=44
)
lgb_model.fit(X, y)
lgb_pred = lgb_model.predict(X_test)

# Create simple ensemble (weighted average)
print("\nCreating ensemble...")
ensemble_pred = 0.4 * rf_pred + 0.35 * xgb_pred + 0.25 * lgb_pred

# Apply natural distribution calibration to maintain zero pattern
calibrated_pred = maintain_natural_distribution(ensemble_pred)

# Force predictions to valid range [0,1] without excessive binarization
calibrated_pred = np.clip(calibrated_pred, 0, 1)

# Generate submission
timestamp = int(time.time())
submission = pd.DataFrame({
    'id': test['id'],
    'rainfall': calibrated_pred
})

# Save submission
os.makedirs('data/candidates', exist_ok=True)
submission_path = f'data/candidates/simple_winner_{timestamp}.csv'
submission.to_csv(submission_path, index=False)

# Print stats
print("\nSubmission statistics:")
print(f"Min: {calibrated_pred.min():.6f}")
print(f"Max: {calibrated_pred.max():.6f}")
print(f"Mean: {calibrated_pred.mean():.6f}")
print(f"Zero count: {(calibrated_pred == 0).sum()} ({(calibrated_pred == 0).mean()*100:.2f}%)")
print(f"Submission saved to {submission_path}")