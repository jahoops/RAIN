import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgbm
from data_pipeline import create_complete_data_pipeline

print("Loading and preparing data...")
X_train_temporal, X_test_temporal = create_complete_data_pipeline()
y_train = X_train_temporal['rainfall']

# Process features
X_train_features = X_train_temporal.drop(['id', 'rainfall'], axis=1)
X_test_features = X_test_temporal.drop(['id'], axis=1)

# Print feature names for reference
print(f"Using {len(X_train_features.columns)} features:")
for i, col in enumerate(X_train_features.columns):
    print(f"{i+1}. {col}")

# Create highly optimized LightGBM with cross-validation
def train_lgbm_with_cv(X, y, X_test, n_folds=5):
    # Optimal parameters based on extensive hyperparameter search
    params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'n_estimators': 10000,
        'learning_rate': 0.007,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'max_depth': 12,
        'lambda_l1': 0.1,
        'lambda_l2': 5,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'feature_fraction': 0.85,
        'verbose': -1
    }
    
    # Initialize arrays for predictions
    oof_predictions = np.zeros(len(X))
    test_predictions = np.zeros(len(X_test))
    
    # Create K-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Train and predict for each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Training fold {fold+1}/{n_folds}...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create dataset for LightGBM
        train_data = lgbm.Dataset(X_train, y_train)
        val_data = lgbm.Dataset(X_val, y_val)
        
        # Train model
        model = lgbm.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            early_stopping_rounds=200,
            verbose_eval=100
        )
        
        # Predict on validation and test
        oof_predictions[val_idx] = model.predict(X_val)
        test_predictions += model.predict(X_test) / n_folds
        
        # Feature importance for this fold
        if fold == 0:
            importance = model.feature_importance(importance_type='gain')
            feature_names = X_train.columns.tolist()
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
            importance_df = importance_df.sort_values('Importance', ascending=False)
            print("\nTop 10 features by importance:")
            print(importance_df.head(10))
    
    return test_predictions, oof_predictions

# Train model with CV
print("\nTraining LightGBM with cross-validation...")
test_preds, oof_preds = train_lgbm_with_cv(X_train_features, y_train, X_test_features)

# Evaluate on training data
from sklearn.metrics import mean_squared_error
oof_mse = mean_squared_error(y_train, oof_preds)
print(f"\nOut-of-fold MSE: {oof_mse:.6f}")
print(f"Out-of-fold score (1-MSE): {1-oof_mse:.6f}")

# Create submission with predictions
submission = pd.DataFrame({
    'id': X_test_temporal['id'],
    'rainfall': test_preds
})

# Save submission
output_path = '/home/USER/Projects/RAIN/data/candidates/optimized_lgbm_cv.csv'
submission.to_csv(output_path, index=False)
print(f"\nCreated optimized LightGBM submission: {output_path}")