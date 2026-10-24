import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgbm
from data_pipeline import create_complete_data_pipeline

# Load data
X_train_temporal, X_test_temporal = create_complete_data_pipeline()
y_train = X_train_temporal['rainfall']

# Process features
X_train = X_train_temporal.drop(['id', 'rainfall'], axis=1)
X_test = X_test_temporal.drop(['id'], axis=1)

# Group features by their patterns
rainfall_cols = ['rainfall_rolling_3d_mean', 'rainfall_lag1', 'rainfall_lag2', 'rainfall_rolling_7d_mean', 'rainfall_lead1']
weather_cols = ['cloud', 'humidity', 'pressure', 'temparature']
wind_cols = ['windspeed', 'winddirection']
time_cols = ['day', 'month']

# Create specialized models for different feature groups
def train_first_layer_models(X_train, y_train, X_test, n_folds=5):
    models = [
        ('rf_rainfall', RandomForestRegressor(n_estimators=300, random_state=42), rainfall_cols),
        ('rf_weather', RandomForestRegressor(n_estimators=300, random_state=43), weather_cols),
        ('lgb_all', lgbm.LGBMRegressor(n_estimators=300, random_state=44), X_train.columns.tolist()),
        ('et_all', ExtraTreesRegressor(n_estimators=300, random_state=45), X_train.columns.tolist())
    ]
    
    # Initialize arrays for meta features
    S_train = np.zeros((X_train.shape[0], len(models)))
    S_test = np.zeros((X_test.shape[0], len(models)))
    
    # Create K-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Train specialized models with cross-validation
    for model_idx, (name, model, cols) in enumerate(models):
        print(f"Training {name} with {len(cols)} features...")
        S_test_i = np.zeros((X_test.shape[0], n_folds))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            print(f"  Fold {fold+1}/{n_folds}")
            X_train_fold, X_val_fold = X_train.iloc[train_idx][cols], X_train.iloc[val_idx][cols]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Train model
            model.fit(X_train_fold, y_train_fold)
            
            # Predict on validation fold and test
            S_train[val_idx, model_idx] = model.predict(X_val_fold)
            S_test_i[:, fold] = model.predict(X_test[cols])
        
        # Average test predictions across folds
        S_test[:, model_idx] = S_test_i.mean(axis=1)
    
    return S_train, S_test

# Train first-layer models
print("Training specialized first-layer models...")
meta_train, meta_test = train_first_layer_models(X_train, y_train, X_test)

# Train second layer model (meta-learner)
print("\nTraining second layer meta-model...")
meta_model = Ridge(alpha=0.01)
meta_model.fit(meta_train, y_train)

print("Meta-model coefficients:")
for i, coef in enumerate(meta_model.coef_):
    print(f"Model {i+1}: {coef:.4f}")

# Generate final predictions
stacked_preds = meta_model.predict(meta_test)

# Load best existing model for blending
best_model = pd.read_csv("/home/USER/Projects/RAIN/data/candidates/submission.85518.csv")
best_preds = best_model.iloc[:, 1].values

# Create final blend: 85% best model + 15% stacked specialized models
final_blend = pd.DataFrame({
    'id': X_test_temporal['id'],
    'rainfall': 0.85 * best_preds + 0.15 * stacked_preds
})

final_blend.to_csv('/home/USER/Projects/RAIN/data/candidates/stacked_specialized_blend.csv', index=False)
print("Created stacked specialized model blend")