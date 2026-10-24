import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel
import os
from optimized_binary_model import create_binary_optimized_features
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def create_optimized_logistic_submission():
    """Create a submission using an ensemble of models with SMOTE"""
    # Load data
    train = pd.read_csv('/home/USER/Projects/RAIN/data/train.csv')
    test = pd.read_csv('/home/USER/Projects/RAIN/data/test.csv')

    # Generate features
    X, X_test = create_binary_optimized_features(train, test)
    y = train['rainfall']
    
    # Create a validation set to estimate performance
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Apply SMOTE only to training data (not validation)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"Training data: {X_train.shape[0]} -> {X_train_smote.shape[0]} after SMOTE")

    preprocessing = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Scale features - fit only on training data
    column_names = X.columns
    preprocessing.fit(X_train_smote)
    
    # Transform all datasets
    X_train_processed = pd.DataFrame(
        preprocessing.transform(X_train_smote), 
        columns=column_names)
    X_val_processed = pd.DataFrame(
        preprocessing.transform(X_val), 
        columns=column_names)
    X_test_processed = pd.DataFrame(
        preprocessing.transform(X_test), 
        columns=column_names)
    
    print(f"Training ensemble models with {X_train_processed.shape[1]} features")
    
    # Create models
    xgb_model = XGBClassifier(
        use_label_encoder=False, 
        eval_metric='logloss',
        objective='binary:logistic',
        random_state=42
    )

    # Train Random Forest with RandomizedSearchCV
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    print("Training Random Forest...")
    rf_random = RandomizedSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42), 
        rf_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, n_iter=10)
    rf_random.fit(X_train_processed, y_train_smote)

    # Train Gradient Boosting with RandomizedSearchCV
    gb_param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }

    print("Training Gradient Boosting...")
    gb_random = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=42), 
        gb_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, n_iter=10)
    gb_random.fit(X_train_processed, y_train_smote)

    # Train XGBoost separately
    print("Training XGBoost...")
    xgb_model.fit(X_train_processed, y_train_smote)

    # Get validation predictions for ensemble weighting
    rf_val_preds = rf_random.best_estimator_.predict_proba(X_val_processed)[:,1]
    gb_val_preds = gb_random.best_estimator_.predict_proba(X_val_processed)[:,1]
    xgb_val_preds = xgb_model.predict_proba(X_val_processed)[:,1]
    
    # Calculate AUC for each model on the validation set
    rf_auc = roc_auc_score(y_val, rf_val_preds)
    gb_auc = roc_auc_score(y_val, gb_val_preds)
    xgb_auc = roc_auc_score(y_val, xgb_val_preds)
    
    print(f"Validation AUC scores:")
    print(f"Random Forest: {rf_auc:.6f}")
    print(f"Gradient Boosting: {gb_auc:.6f}")
    print(f"XGBoost: {xgb_auc:.6f}")
    
    # Weight models by their performance (normalized)
    total_auc = rf_auc + gb_auc + xgb_auc
    rf_weight = rf_auc / total_auc
    gb_weight = gb_auc / total_auc
    xgb_weight = xgb_auc / total_auc
    
    print(f"Model weights:")
    print(f"Random Forest: {rf_weight:.4f}")
    print(f"Gradient Boosting: {gb_weight:.4f}")
    print(f"XGBoost: {xgb_weight:.4f}")
    
    # Get test predictions
    rf_preds = rf_random.best_estimator_.predict_proba(X_test_processed)[:,1]
    gb_preds = gb_random.best_estimator_.predict_proba(X_test_processed)[:,1]
    xgb_preds = xgb_model.predict_proba(X_test_processed)[:,1]
    
    # Weighted average of predictions
    ensemble_preds = (
        rf_weight * rf_preds + 
        gb_weight * gb_preds + 
        xgb_weight * xgb_preds
    )
    
    # Calculate ensemble AUC on validation set
    ensemble_val_preds = (
        rf_weight * rf_val_preds + 
        gb_weight * gb_val_preds + 
        xgb_weight * xgb_val_preds
    )
    ensemble_auc = roc_auc_score(y_val, ensemble_val_preds)
    print(f"Ensemble validation AUC: {ensemble_auc:.6f}")
    
    # Create submission
    submission = pd.DataFrame({
        'id': test['id'],
        'rainfall': ensemble_preds
    })
    
    output_path = f"data/candidates/weighted_ensemble_auc_{ensemble_auc:.6f}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    # Print best parameters
    print("\nBest Random Forest parameters:")
    print(rf_random.best_params_)
    print("\nBest Gradient Boosting parameters:")
    print(gb_random.best_params_)

if __name__ == "__main__":
    create_optimized_logistic_submission()