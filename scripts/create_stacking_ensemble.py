import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgbm
from data_pipeline import create_complete_data_pipeline

print("Loading and preparing data...")
X_train_temporal, X_test_temporal = create_complete_data_pipeline()
y_train = X_train_temporal['rainfall']

# Process features
X_train_features = X_train_temporal.drop(['id', 'rainfall'], axis=1)
X_test_features = X_test_temporal.drop(['id'], axis=1)

# Define base models
def create_base_models():
    models = [
        ('rf', RandomForestRegressor(
            n_estimators=300, 
            max_depth=None, 
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=0.7,
            random_state=42,
            n_jobs=-1)),
        ('et', ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1, 
            max_features=0.8,
            random_state=43,
            n_jobs=-1)),
        ('lgb', lgbm.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.01,
            num_leaves=31,
            max_depth=12,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=5,
            random_state=44))
    ]
    return models

# Create stacked predictions with cross-validation
def stacking_predictions(X, y, X_test, models, n_folds=5):
    # Initialize arrays for meta features
    S_train = np.zeros((X.shape[0], len(models)))
    S_test = np.zeros((X_test.shape[0], len(models)))
    
    # Create K-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Generate out-of-fold predictions for training data and full predictions for test data
    for model_idx, (name, model) in enumerate(models):
        print(f"Training {name} with cross-validation...")
        S_test_i = np.zeros((X_test.shape[0], n_folds))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"  Fold {fold+1}/{n_folds}")
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model.fit(X_train_fold, y_train_fold)
            
            # Predict on validation fold and test
            S_train[val_idx, model_idx] = model.predict(X_val_fold)
            S_test_i[:, fold] = model.predict(X_test)
        
        # Average test predictions across folds
        S_test[:, model_idx] = S_test_i.mean(axis=1)
    
    return S_train, S_test

# Fit meta-model
def fit_meta_model(S_train, y, S_test):
    print("\nTraining meta-model...")
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(S_train, y)
    
    # Print coefficients (model weights)
    print("Meta-model coefficients (weights):")
    for i, coef in enumerate(meta_model.coef_):
        print(f"Model {i+1}: {coef:.4f}")
    
    # Generate final predictions
    final_predictions = meta_model.predict(S_test)
    return final_predictions

# Run stacking procedure
base_models = create_base_models()
S_train, S_test = stacking_predictions(X_train_features, y_train, X_test_features, base_models)
final_preds = fit_meta_model(S_train, y_train, S_test)

# Create submission
submission = pd.DataFrame({
    'id': X_test_temporal['id'],
    'rainfall': final_preds
})

# Save submission
output_path = '/home/USER/Projects/RAIN/data/candidates/stacking_ensemble.csv'
submission.to_csv(output_path, index=False)
print(f"\nCreated stacking ensemble submission: {output_path}")