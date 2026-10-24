import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import time

def enhance_with_temporal_features(X_train, y_train, X_test):
    """Add the optimal temporal features discovered by the GA"""
    print("Enhancing with temporal features...")
    
    # Create copies to avoid modifying originals
    X_train_enhanced = X_train.copy()
    X_test_enhanced = X_test.copy()
    
    # 1. Create lag features (most important temporal features)
    print("Creating lag features...")
    if 'rainfall' in X_train:
        for lag in [1, 2, 3, 7]:
            X_train_enhanced[f'rainfall_lag_{lag}'] = X_train['rainfall'].shift(lag)
            
        # Fill missing values at the beginning
        for col in X_train_enhanced.columns:
            if 'lag' in col and X_train_enhanced[col].isna().sum() > 0:
                X_train_enhanced[col] = X_train_enhanced[col].fillna(0)
    
    # 2. Create rolling window features (second most important)
    print("Creating rolling statistics...")
    if 'rainfall' in X_train:
        for window in [3, 7, 14]:
            # Mean (most important)
            X_train_enhanced[f'rainfall_roll_mean_{window}'] = X_train['rainfall'].rolling(window).mean()
            # Max (captures extreme events)
            X_train_enhanced[f'rainfall_roll_max_{window}'] = X_train['rainfall'].rolling(window).max()
            # Standard deviation (captures volatility)
            X_train_enhanced[f'rainfall_roll_std_{window}'] = X_train['rainfall'].rolling(window).std()
            
        # Fill missing values
        for col in X_train_enhanced.columns:
            if 'roll' in col and X_train_enhanced[col].isna().sum() > 0:
                X_train_enhanced[col] = X_train_enhanced[col].fillna(0)
    
    # 3. Create cloud interaction features (most important predictor)
    print("Creating cloud interactions...")
    if 'cloud' in X_train_enhanced.columns:
        # Cloud squared (non-linear effect)
        X_train_enhanced['cloud_squared'] = X_train_enhanced['cloud'] ** 2
        X_test_enhanced['cloud_squared'] = X_test_enhanced['cloud'] ** 2
        
        # Cloud × temperature interaction
        if 'temperature' in X_train_enhanced.columns:
            X_train_enhanced['cloud_temp'] = X_train_enhanced['cloud'] * X_train_enhanced['temperature']
            X_test_enhanced['cloud_temp'] = X_test_enhanced['cloud'] * X_test_enhanced['temperature']
        
        # Cloud × humidity interaction
        if 'humidity' in X_train_enhanced.columns:
            X_train_enhanced['cloud_humidity'] = X_train_enhanced['cloud'] * X_train_enhanced['humidity']
            X_test_enhanced['cloud_humidity'] = X_test_enhanced['cloud'] * X_test_enhanced['humidity']
    
    # Add all missing features to test data with appropriate defaults
    print("Aligning test features...")
    missing_cols = set(X_train_enhanced.columns) - set(X_test_enhanced.columns) - {'rainfall'}
    
    for col in missing_cols:
        if 'lag' in col or 'roll' in col:
            # For lag/rolling features based on rainfall, use zeros
            X_test_enhanced[col] = 0
        else:
            # For other features, use median from training
            if col in X_train_enhanced.columns:
                X_test_enhanced[col] = X_train_enhanced[col].median()
            else:
                X_test_enhanced[col] = 0
    
    print(f"Enhanced features: {X_train_enhanced.shape[1]} total columns")
    return X_train_enhanced, X_test_enhanced

def create_specialized_models(X_train, y_train, X_test):
    """Create specialized models focusing on different feature groups"""
    models = {}
    predictions = {}
    
    # 1. Cloud-focused model
    print("\nTraining cloud-focused model...")
    cloud_features = [col for col in X_train.columns if 
                     any(x in col for x in ['cloud', 'Cloud'])]
    
    # Add some basic weather features to support cloud data
    basic_weather = ['temperature', 'humidity', 'pressure']
    for feature in basic_weather:
        candidates = [col for col in X_train.columns if feature in col.lower()]
        cloud_features.extend(candidates[:1])  # Add at most one of each type
        
    if cloud_features:
        print(f"Cloud model features ({len(cloud_features)}): {', '.join(cloud_features[:5])}...")
        cloud_model = RandomForestRegressor(n_estimators=500, max_depth=None, 
                                           min_samples_leaf=2, n_jobs=-1, random_state=42)
        cloud_model.fit(X_train[cloud_features], y_train)
        cloud_preds = cloud_model.predict(X_test[cloud_features])
        models['cloud'] = cloud_model
        predictions['cloud'] = cloud_preds
        
        # Evaluate on training data
        try:
            cloud_cv = cross_val_score(cloud_model, X_train[cloud_features], y_train, 
                                      cv=5, scoring='neg_mean_squared_error')
            print(f"Cloud model CV MSE: {-cloud_cv.mean():.6f}")
        except Exception as e:
            print(f"Cloud model CV error: {e}")
    
    # 2. Temporal-focused model
    print("\nTraining temporal-focused model...")
    temporal_features = [col for col in X_train.columns if 
                        any(x in col for x in ['lag', 'roll', 'trend'])]
    
    if temporal_features:
        print(f"Temporal model features ({len(temporal_features)}): {', '.join(temporal_features[:5])}...")
        temporal_model = GradientBoostingRegressor(n_estimators=300, max_depth=5, 
                                                  learning_rate=0.05, random_state=42)
        temporal_model.fit(X_train[temporal_features], y_train)
        temporal_preds = temporal_model.predict(X_test[temporal_features])
        models['temporal'] = temporal_model
        predictions['temporal'] = temporal_preds
        
        # Evaluate on training data
        try:
            temporal_cv = cross_val_score(temporal_model, X_train[temporal_features], y_train, 
                                         cv=5, scoring='neg_mean_squared_error')
            print(f"Temporal model CV MSE: {-temporal_cv.mean():.6f}")
        except Exception as e:
            print(f"Temporal model CV error: {e}")
    
    # 3. Combined model with all features
    print("\nTraining combined model with all features...")
    combined_model = xgb.XGBRegressor(n_estimators=400, max_depth=6, 
                                     learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, random_state=42)
    combined_model.fit(X_train, y_train)
    combined_preds = combined_model.predict(X_test)
    models['combined'] = combined_model
    predictions['combined'] = combined_preds
    
    # Evaluate on training data using XGBoost's built-in CV or a manual split
    try:
        # Try using built-in XGBoost evaluation
        print("Using XGBoost's built-in evaluation...")
        X_train_val, X_val, y_train_val, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        val_model = xgb.XGBRegressor(n_estimators=400, max_depth=6, 
                                   learning_rate=0.05, subsample=0.8,
                                   colsample_bytree=0.8, random_state=42)
        val_model.fit(X_train_val, y_train_val)
        val_preds = val_model.predict(X_val)
        val_mse = mean_squared_error(y_val, val_preds)
        print(f"Combined model validation MSE: {val_mse:.6f}")
    except Exception as e:
        print(f"Combined model evaluation error: {e}")
    
    return models, predictions

def create_ensemble_prediction(predictions):
    """Create ensemble prediction with optimal weights"""
    # Use weights proportional to feature importance from GA
    # Cloud ~38%, Temporal ~60%, Combined for stability
    if 'cloud' in predictions and 'temporal' in predictions:
        print("\nCreating weighted ensemble prediction...")
        ensemble_pred = (
            0.35 * predictions['cloud'] +
            0.45 * predictions['temporal'] +
            0.20 * predictions['combined']
        )
    else:
        print("\nFalling back to combined model only...")
        ensemble_pred = predictions['combined']
    
    return ensemble_pred

if __name__ == "__main__":
    # Import data pipeline
    from data_pipeline import create_complete_data_pipeline
    
    # Load data
    print("Loading data...")
    X_train_temporal, X_test_temporal = create_complete_data_pipeline()
    y_train = X_train_temporal['rainfall']
    X_train = X_train_temporal.drop(['id', 'rainfall'], axis=1)
    X_test = X_test_temporal.drop(['id'], axis=1)
    
    # Add temporal features
    X_train_enhanced, X_test_enhanced = enhance_with_temporal_features(
        X_train_temporal, y_train, X_test_temporal
    )
    
    # Remove id and rainfall from enhanced data
    if 'id' in X_train_enhanced.columns:
        X_train_enhanced = X_train_enhanced.drop('id', axis=1)
    if 'rainfall' in X_train_enhanced.columns:
        X_train_enhanced = X_train_enhanced.drop('rainfall', axis=1)
    
    if 'id' in X_test_enhanced.columns:
        X_test_enhanced = X_test_enhanced.drop('id', axis=1)
    if 'rainfall' in X_test_enhanced.columns:
        X_test_enhanced = X_test_enhanced.drop('rainfall', axis=1)
    
    # Create specialized models
    models, predictions = create_specialized_models(X_train_enhanced, y_train, X_test_enhanced)
    
    # Create ensemble prediction
    ensemble_pred = create_ensemble_prediction(predictions)
    
    # Create submission
    submission = pd.DataFrame({
        'id': X_test_temporal['id'],
        'rainfall': ensemble_pred
    })
    
    # Save the submission
    timestamp = int(time.time())
    submission_path = f'/home/USER/Projects/RAIN/data/candidates/ga_ensemble_{timestamp}.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Created ensemble submission at {submission_path}")
    
    # Basic statistics
    print("\nPrediction statistics:")
    print(f"Min: {ensemble_pred.min():.6f}")
    print(f"Max: {ensemble_pred.max():.6f}")
    print(f"Mean: {ensemble_pred.mean():.6f}")
    print(f"Zeros: {(ensemble_pred == 0).sum()} ({(ensemble_pred == 0).mean()*100:.2f}%)")