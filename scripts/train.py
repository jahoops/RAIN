import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import roc_auc_score, mean_absolute_error
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_data, preprocess_data

def train_model():
    print("Loading and preprocessing data...")
    train_df, test_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # Prepare features and target
    X = train_df.drop(['id', 'rainfall'], axis=1, errors='ignore')
    y = train_df['rainfall']
    
    # Create train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train classification model
    print("Training model...")
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        early_stopping_rounds=30,
        verbose=True
    )
    
    # Save the model and feature names
    os.makedirs('models', exist_ok=True)
    pickle.dump(model, open('models/model.pkl', 'wb'))
    pd.DataFrame({'feature': X.columns}).to_csv('models/feature_names.csv', index=False)
    
    # Get validation predictions and calculate AUC
    val_preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_preds)
    print(f"Validation AUC: {auc:.4f}")
    
    # Generate predictions for test data
    test_preds = model.predict_proba(test_df.drop(['id'], axis=1, errors='ignore'))[:, 1]
    
    # Create submission with only id and rainfall prediction
    submission = pd.DataFrame({
        'id': test_df['id'],
        'rainfall': test_preds
    })
    submission.to_csv('data/submission.csv', index=False)
    print("Predictions saved to 'data/submission.csv'")

if __name__ == "__main__":
    train_model()
