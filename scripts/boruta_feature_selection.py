import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
import pickle
import os
import time

from data_loader import load_data, preprocess_data
from feature_engineering import engineer_features
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
from temporal_features import add_temporal_rainfall_features

def run_boruta_feature_selection():
    """
    Apply Boruta feature selection to the training data and compare with current features
    """
    print("Loading and preprocessing data...")
    # Load and preprocess data
    train_df, test_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df)
    train_df, test_df = add_temporal_rainfall_features(train_df, test_df)
    # Apply feature engineering
    train_df, test_df = engineer_features(train_df, test_df)
    
    # Remove categorical columns that have been one-hot encoded
    if 'season' in train_df.columns:
        train_df = train_df.drop('season', axis=1)
    if 'season' in test_df.columns:
        test_df = test_df.drop('season', axis=1)
    
    # Prepare features and target
    X = train_df.drop(['id', 'rainfall'], axis=1, errors='ignore')
    y = train_df['rainfall']
    
    # Create directory for output
    os.makedirs('models/feature_selection', exist_ok=True)
    
    # Save original feature list for comparison
    original_features = pd.DataFrame({'feature': X.columns})
    original_features.to_csv('models/feature_selection/original_features.csv', index=False)
    print(f"Original feature count: {len(X.columns)}")
    
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and run Boruta feature selection
    print("Running Boruta feature selection (this may take some time)...")
    start_time = time.time()
    
    # Initialize Random Forest for Boruta
    rf = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=42)
    
    # Initialize Boruta
    boruta = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42)
    
    # Fit Boruta (it accepts numpy arrays, not pandas DataFrames)
    boruta.fit(X_train.values, y_train.values)
    
    # Get selected features
    selected_features = X.columns[boruta.support_]
    tentative_features = X.columns[boruta.support_weak_]
    
    # Print results
    print(f"\nBoruta feature selection completed in {(time.time() - start_time)/60:.2f} minutes")
    print(f"Selected features: {len(selected_features)} out of {len(X.columns)}")
    print(f"Tentative features: {len(tentative_features)}")
    
    # Save selected features
    pd.DataFrame({'feature': selected_features}).to_csv(
        'models/feature_selection/boruta_selected_features.csv', index=False)
    pd.DataFrame({'feature': tentative_features}).to_csv(
        'models/feature_selection/boruta_tentative_features.csv', index=False)
    
    print("\nTop selected features:")
    for feature in selected_features[:20]:  # Show top 20
        print(f"- {feature}")
    
    # Compare model performance with original vs selected features
    print("\nComparing model performance...")
    
    # Train and evaluate with all features
    full_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    full_model.fit(X_train, y_train)
    full_preds = full_model.predict(X_val)
    full_score = roc_auc_score(y_val, full_preds)
    
    # Train and evaluate with Boruta selected features
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    
    boruta_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    boruta_model.fit(X_train_selected, y_train)
    boruta_preds = boruta_model.predict(X_val_selected)
    boruta_score = roc_auc_score(y_val, boruta_preds)
    
    # Print comparison
    print(f"\nPerformance comparison:")
    print(f"All features ({len(X.columns)}): AUC = {full_score:.6f}")
    print(f"Boruta selected features ({len(selected_features)}): AUC = {boruta_score:.6f}")
    
    # Plot feature importance comparison
    plt.figure(figsize=(12, 10))
    
    # Get feature importance for selected features
    feature_importances = boruta_model.feature_importances_
    sorted_idx = np.argsort(feature_importances)
    
    # Plot
    plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [selected_features[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Boruta Selected Features - Importance')
    plt.tight_layout()
    plt.savefig('models/feature_selection/boruta_feature_importance.png')
    
    return selected_features, full_score, boruta_score

if __name__ == "__main__":
    selected_features, full_score, boruta_score = run_boruta_feature_selection()
    print("\nBoruta feature selection completed.")
