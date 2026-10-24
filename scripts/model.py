import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import pickle
import os

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, 
                 params: Dict[str, Any] = None) -> xgb.XGBClassifier:
    """
    Train an XGBoost model
    
    Args:
        X_train: Training features
        y_train: Target variable
        params: XGBoost parameters
    
    Returns:
        Trained XGBoost model
    """
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'eta': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'tree_method': 'hist',
            'n_estimators': 500
        }
    
    # Split data for validation
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, 
                                                test_size=0.2, random_state=42)
    
    # Train model
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=50
    )
    
    # Evaluate
    val_preds = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, val_preds)
    print(f"Validation AUC: {auc_score:.4f}")
    
    return model

def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, 
                  params: Dict[str, Any] = None) -> lgb.LGBMClassifier:
    """
    Train a LightGBM model
    
    Args:
        X_train: Training features
        y_train: Target variable
        params: LightGBM parameters
    
    Returns:
        Trained LightGBM model
    """
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'n_estimators': 500
        }
    
    # Split data for validation
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, 
                                                test_size=0.2, random_state=42)
    
    # Train model
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=50
    )
    
    # Evaluate
    val_preds = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, val_preds)
    print(f"Validation AUC: {auc_score:.4f}")
    
    return model

def save_model(model, filename: str) -> None:
    """
    Save model to disk
    
    Args:
        model: Trained model
        filename: Output filename
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def load_model(filename: str) -> Any:
    """
    Load model from disk
    
    Args:
        filename: Model filename
    
    Returns:
        Loaded model
    """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model
