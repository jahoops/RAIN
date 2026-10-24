import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import time
import pickle
import json

# Configuration
RESULTS_DIR = "feature_evaluation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/plots", exist_ok=True)
N_FOLDS = 5
RANDOM_STATE = 42

# RF model configuration
RF_CONFIG = {
    'n_estimators': 1000,
    'max_depth': 12,
    'min_samples_split': 3,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'bootstrap': True,
    'n_jobs': -1,
    'random_state': RANDOM_STATE
}

def basic_features(train, test):
    """Basic feature set with minimal processing"""
    X = train.drop(['id', 'rainfall'], axis=1).copy()
    if test is not None:
        X_test = test.drop(['id'], axis=1).copy()
    else:
        X_test = None
    
    # No new features
    new_features = []
    
    return X, X_test, new_features

def temporal_features(train, test):
    """Add temporal/seasonal features"""
    X = train.copy()
    if test is not None:
        X_test = test.copy()
    else:
        X_test = None
    
    new_features = []
    
    # Create month from ID (assuming IDs relate to days in sequence)
    if 'id' in X.columns:
        X['month'] = ((X['id'] % 365) // 30) + 1
        new_features.append('month')
        
        # Season (1-4)
        X['season'] = ((X['month'] - 1) // 3) + 1
        new_features.append('season')
        
        # Day of year
        X['day_of_year'] = X['id'] % 365
        new_features.append('day_of_year')
        
        # Cyclic encoding of day of year
        X['day_sin'] = np.sin(2 * np.pi * X['day_of_year']/365)
        new_features.append('day_sin')
        
        X['day_cos'] = np.cos(2 * np.pi * X['day_of_year']/365)
        new_features.append('day_cos')
        
        if X_test is not None:
            X_test['month'] = ((X_test['id'] % 365) // 30) + 1
            X_test['season'] = ((X_test['month'] - 1) // 3) + 1
            X_test['day_of_year'] = X_test['id'] % 365
            X_test['day_sin'] = np.sin(2 * np.pi * X_test['day_of_year']/365)
            X_test['day_cos'] = np.cos(2 * np.pi * X_test['day_of_year']/365)
    
    # Drop target and ID for modeling
    X_final = X.drop(['id', 'rainfall'], axis=1)
    if X_test is not None:
        X_test_final = X_test.drop(['id'], axis=1)
    else:
        X_test_final = None
    
    return X_final, X_test_final, new_features

def statistical_features(train, test):
    """Apply statistical transformations to features"""
    X = train.copy()
    if test is not None:
        X_test = test.copy()
    else:
        X_test = None
    
    # Get numerical columns excluding ID and target
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    if 'id' in num_cols:
        num_cols.remove('id')
    if 'rainfall' in num_cols:
        num_cols.remove('rainfall')
        
    new_features = []
    
    # Add temporal features
    if 'id' in X.columns:
        X['month'] = ((X['id'] % 365) // 30) + 1
        new_features.append('month')
        
        # Season (1-4)
        X['season'] = ((X['month'] - 1) // 3) + 1
        new_features.append('season')
        
        if X_test is not None:
            X_test['month'] = ((X_test['id'] % 365) // 30) + 1
            X_test['season'] = ((X_test['month'] - 1) // 3) + 1
    
    # Square and square root transformations
    for col in num_cols[:10]:  # Limit to top 10 features to avoid dimensionality explosion
        X[f'{col}_squared'] = X[col] ** 2
        new_features.append(f'{col}_squared')
        
        # Square root for non-negative values
        if (X[col] >= 0).all():
            X[f'{col}_sqrt'] = np.sqrt(X[col])
            new_features.append(f'{col}_sqrt')
            
        if X_test is not None:
            X_test[f'{col}_squared'] = X_test[col] ** 2
            if (X[col] >= 0).all():
                X_test[f'{col}_sqrt'] = np.sqrt(X_test[col])
    
    # Feature interactions
    for i in range(min(5, len(num_cols))):
        for j in range(i+1, min(6, len(num_cols))):
            col1 = num_cols[i]
            col2 = num_cols[j]
            
            X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
            new_features.append(f'{col1}_x_{col2}')
            
            if X_test is not None:
                X_test[f'{col1}_x_{col2}'] = X_test[col1] * X_test[col2]
    
    # Drop target and ID for modeling
    X_final = X.drop(['id', 'rainfall'], axis=1)
    if X_test is not None:
        X_test_final = X_test.drop(['id'], axis=1)
    else:
        X_test_final = None
    
    # Handle any missing values
    X_final = X_final.fillna(X_final.mean())
    if X_test_final is not None:
        X_test_final = X_test_final.fillna(X_final.mean())
    
    return X_final, X_test_final, new_features

def combined_features(train, test):
    """Combine multiple feature sets"""
    # Start with basic features
    X, X_test, basic_new = basic_features(train, test)
    
    # Add temporal features
    temp_X, temp_X_test, temp_new = temporal_features(train, test)
    
    # Add statistical features
    stat_X, stat_X_test, stat_new = statistical_features(train, test)
    
    # Combine all features
    all_cols = list(set(X.columns).union(set(temp_X.columns)).union(set(stat_X.columns)))
    
    # Create final dataframes
    X_final = pd.DataFrame(index=X.index)
    for col in all_cols:
        if col in X.columns:
            X_final[col] = X[col]
        elif col in temp_X.columns:
            X_final[col] = temp_X[col]
        elif col in stat_X.columns:
            X_final[col] = stat_X[col]
    
    # Handle test data if provided
    if X_test is not None and temp_X_test is not None and stat_X_test is not None:
        X_test_final = pd.DataFrame(index=X_test.index)
        for col in all_cols:
            if col in X_test.columns:
                X_test_final[col] = X_test[col]
            elif col in temp_X_test.columns:
                X_test_final[col] = temp_X_test[col]
            elif col in stat_X_test.columns:
                X_test_final[col] = stat_X_test[col]
            else:
                # If column exists in training but not test, add it with zeros
                X_test_final[col] = 0
    else:
        X_test_final = None
    
    # Combine all new features
    all_new_features = basic_new + temp_new + stat_new
    
    # Handle any missing values
    X_final = X_final.fillna(X_final.mean())
    if X_test_final is not None:
        X_test_final = X_test_final.fillna(X_final.mean())
    
    return X_final, X_test_final, all_new_features

def evaluate_feature_set(name, feature_func):
    """Evaluate a feature set using cross-validation with detailed metrics including AUC"""
    print(f"\n{'-'*50}")
    print(f"Evaluating feature set: {name}")
    print(f"{'-'*50}")
    
    start_time = time.time()
    
    # Load data
    train = pd.read_csv('/home/USER/Projects/RAIN/data/train.csv')
    
    # Generate features - only train, not test
    try:
        X, _, new_features = feature_func(train, None)
        
        # Output feature information
        print(f"Generated {len(X.columns)} total features")
        if new_features:
            print(f"Added {len(new_features)} new features")
        
        # Initialize model
        rf = RandomForestRegressor(**RF_CONFIG)
        
        # Target variable
        y = train['rainfall']
        
        # Perform cross-validation for predictions
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_predictions = cross_val_predict(rf, X, y, cv=kf)
        
        # Calculate AUC (using R² as AUC equivalent)
        auc = r2_score(y, cv_predictions)
        
        # Calculate additional metrics for reference
        rmse = np.sqrt(mean_squared_error(y, cv_predictions))
        mae = mean_absolute_error(y, cv_predictions)
        
        # Calculate cross-validation AUC scores
        auc_scores = cross_val_score(rf, X, y, cv=kf, scoring='r2')
        
        # Calculate absolute errors for each prediction
        errors = np.abs(cv_predictions - y)
        
        # Create a DataFrame with predictions, actual values, and errors
        results_df = pd.DataFrame({
            'actual': y,
            'predicted': cv_predictions,
            'error': cv_predictions - y,
            'abs_error': errors
        })
        
        # Save the results DataFrame
        results_df.to_csv(f"{RESULTS_DIR}/{name}_predictions.csv", index=False)
        
        # Train final model on all data and get feature importances
        rf.fit(X, y)
        feature_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save feature importances
        feature_importances.to_csv(f"{RESULTS_DIR}/{name}_importances.csv", index=False)
        
        # Save the model
        with open(f"{RESULTS_DIR}/{name}_model.pkl", 'wb') as f:
            pickle.dump(rf, f)
        
        # Create performance summary
        summary = {
            "name": name,
            "auc": float(auc),                     # Changed RMSE to AUC
            "auc_mean": float(auc_scores.mean()),  # Added mean AUC from CV
            "auc_std": float(auc_scores.std()),    # Added AUC standard deviation
            "rmse": float(rmse),                   # Keep RMSE for reference
            "mae": float(mae),                     # Keep MAE for reference
            "mean_abs_error": float(errors.mean()),
            "median_abs_error": float(np.median(errors)),
            "max_abs_error": float(errors.max()),
            "feature_count": len(X.columns),
            "top_features": feature_importances.head(10).to_dict(orient='records'),
            "execution_time": time.time() - start_time
        }
        
        # Save summary to JSON
        with open(f"{RESULTS_DIR}/{name}_summary.json", 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Create plots
        
        # 1. Actual vs Predicted scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y, cv_predictions, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{name}: Actual vs Predicted (AUC = {auc:.4f}, RMSE = {rmse:.4f})')
        plt.savefig(f"{RESULTS_DIR}/plots/{name}_actual_vs_predicted.png")
        plt.close()
        
        # 2. Error distribution
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50)
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.title(f'{name}: Error Distribution (Mean = {errors.mean():.4f})')
        plt.savefig(f"{RESULTS_DIR}/plots/{name}_error_dist.png")
        plt.close()
        
        # 3. Prediction errors vs actual values
        plt.figure(figsize=(10, 6))
        plt.scatter(y, errors, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Actual Value')
        plt.ylabel('Prediction Error')
        plt.title(f'{name}: Error vs Actual Value')
        plt.savefig(f"{RESULTS_DIR}/plots/{name}_error_vs_actual.png")
        plt.close()
        
        # 4. Top feature importances
        plt.figure(figsize=(12, 8))
        top_n = feature_importances.head(20)
        plt.barh(top_n['feature'][::-1], top_n['importance'][::-1])
        plt.xlabel('Importance')
        plt.title(f'{name}: Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/plots/{name}_feature_importance.png")
        plt.close()
        
        print(f"Evaluation of {name} completed in {time.time() - start_time:.2f}s")
        print(f"AUC: {auc:.4f} (±{auc_scores.std():.4f}), RMSE: {rmse:.4f}")
        print(f"Results saved to {RESULTS_DIR}/{name}_summary.json")
        
        return summary
        
    except Exception as e:
        print(f"Error evaluating {name}: {e}")
        return None

def compare_all_features():
    """Evaluate and compare all feature sets"""
    feature_sets = {
        "basic": basic_features,
        "temporal": temporal_features,
        "statistical": statistical_features,
        "combined": combined_features,
    }
    
    results = []
    
    # Evaluate each feature set
    for name, func in feature_sets.items():
        summary = evaluate_feature_set(name, func)
        if summary:
            results.append(summary)
    
    if not results:
        print("No feature sets were successfully evaluated")
        return None
    
    # Create comparison table - now sort by AUC (higher is better)
    comparison = pd.DataFrame(results)
    comparison = comparison[['name', 'auc', 'rmse', 'mae', 'feature_count', 'execution_time']]
    comparison = comparison.sort_values('auc', ascending=False)  # Changed from rmse to auc and reversed sort order
    
    # Save comparison
    comparison.to_csv(f"{RESULTS_DIR}/feature_comparison.csv", index=False)
    
    # Create comparison plot
    plt.figure(figsize=(12, 6))
    plt.barh(comparison['name'], comparison['auc'], color='blue', alpha=0.6, label='AUC (R²)')
    plt.barh(comparison['name'], -comparison['rmse'], color='red', alpha=0.6, label='RMSE (negative)')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Metric Value')
    plt.title('Feature Set Performance Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/plots/feature_comparison.png")
    
    print("\nFeature Set Comparison:")
    print(comparison.to_string(index=False))
    print(f"\nComparison saved to {RESULTS_DIR}/feature_comparison.csv")
    
    # Create actual submissions for the top performing feature set
    best_feature_set = comparison.iloc[0]['name']
    create_submission(best_feature_set, feature_sets[best_feature_set])
    
    return comparison

def create_submission(name, feature_func):
    """Create a submission file using the specified feature set with AUC score in filename"""
    print(f"\nCreating submission for best feature set: {name}")
    
    # Load data
    train = pd.read_csv('/home/USER/Projects/RAIN/data/train.csv')
    test = pd.read_csv('/home/USER/Projects/RAIN/data/test.csv')
    
    # Generate features
    X, X_test, _ = feature_func(train, test)
    y = train['rainfall']
    
    # Calculate cross-validation AUC score
    print("Calculating cross-validation AUC (R²) score...")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    auc_scores = cross_val_score(RandomForestRegressor(**RF_CONFIG), X, y, cv=kf, scoring='r2')
    auc = auc_scores.mean()
    print(f"Cross-validation AUC: {auc:.6f}")
    
    # Train model
    rf = RandomForestRegressor(**RF_CONFIG)
    rf.fit(X, y)
    
    # Generate predictions
    predictions = rf.predict(X_test)
    
    # Create submission with AUC score in filename
    auc_str = f"auc_{auc:.6f}"
    output_path = f"data/candidates/rf_{name}_{auc_str}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    submission = pd.DataFrame({
        'id': test['id'],
        'rainfall': predictions
    })
    
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    # Get feature importances
    indices = rf.feature_importances_.argsort()[::-1][:20]
    importances = rf.feature_importances_[indices]
    
    # Create model documentation - ensure all values are JSON serializable
    model_doc = {
        "name": name,
        "auc": float(auc),  # Changed from "score" to "auc"
        "feature_count": int(len(X.columns)),
        "model_params": RF_CONFIG,
        "features": list(X.columns),
        "top_features": [
            {
                "feature": str(X.columns[i]), 
                "importance": float(imp)
            }
            for i, imp in zip(indices, importances)
        ]
    }
    
    with open(f"{RESULTS_DIR}/{name}_{auc_str}_model.json", 'w') as f:
        json.dump(model_doc, f, indent=4)

if __name__ == "__main__":
    compare_all_features()