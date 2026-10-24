import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split  # Added train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from data_loader import load_data, preprocess_data
from feature_engineering import engineer_features

def analyze_validation_gap():
    """Analyze and visualize the gap between CV scores and submission scores"""
    # Load previous results if available
    try:
        results_df = pd.read_csv('/home/USER/Projects/RAIN/models/experiment_results.csv')
        print("Loaded experiment results:")
        print(results_df.head())
        
        # Plot CV scores vs actual submission scores if available
        if 'CV Score' in results_df.columns and 'Submission Score' in results_df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(results_df['CV Score'], results_df['Submission Score'])
            plt.plot([0.5, 1], [0.5, 1], 'r--')  # Diagonal line for reference
            plt.xlabel('Cross-Validation Score')
            plt.ylabel('Actual Submission Score')
            plt.title('CV Score vs. Submission Score')
            plt.savefig('/home/USER/Projects/RAIN/plots/cv_vs_submission.png')
            print("CV vs. submission score plot saved")
    except:
        print("No existing results found to analyze")

def perform_time_based_validation():
    """Implement time-based validation to better simulate the test set evaluation"""
    print("Performing time-based validation...")
    
    # Load and preprocess data
    train_df, test_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # Check if 'day' column exists for time-based validation
    if 'day' not in train_df.columns:
        print("No 'day' column found for time-based validation")
        return
    
    # Sort by day
    train_df = train_df.sort_values('day')
    
    # Get target and features
    X = train_df.drop(['id', 'rainfall'], axis=1, errors='ignore')
    y = train_df['rainfall']
    
    # Implement time-series split with 5 folds
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Try all feature strategies
    strategies = ['all', 'minimal', 'rainfall_patterns', 'all_with_patterns']
    
    results = []
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        
        # Apply feature engineering
        train_engineered, _ = engineer_features(train_df, test_df, strategy=strategy)
        X = train_engineered.drop(['id', 'rainfall'], axis=1, errors='ignore')
        
        fold_scores = []
        
        # Perform time-based cross-validation
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            # Ensure indices align with sorted dataframe
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train a simple model
            import xgboost as xgb
            model = xgb.XGBRegressor(
                n_estimators=200, 
                learning_rate=0.05,
                max_depth=4,  # Reduced complexity
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=2.0,  # Increased regularization
                reg_lambda=2.0,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Make predictions
            val_preds = model.predict(X_val)
            
            # Calculate score
            score = roc_auc_score(y_val, val_preds)
            fold_scores.append(score)
            print(f"Fold {fold+1} score: {score:.6f}")
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        print(f"Mean time-based CV score: {mean_score:.6f} (±{std_score:.6f})")
        
        results.append({
            'Strategy': strategy,
            'Mean CV Score': mean_score,
            'Std CV Score': std_score
        })
    
    # Create summary table
    results_df = pd.DataFrame(results)
    print("\nTime-based validation results:")
    print(results_df)
    
    # Save results
    results_df.to_csv('/home/USER/Projects/RAIN/models/time_based_validation_results.csv', index=False)
    print("Results saved to '/home/USER/Projects/RAIN/models/time_based_validation_results.csv'")
    
    return results_df

def check_for_data_leakage():
    """Check for potential data leakage in the feature engineering process"""
    print("Checking for data leakage...")
    
    # Load raw data
    train_df, test_df = load_data()
    
    # Check if 'day' column exists and is sorted
    if 'day' in train_df.columns:
        day_diff = train_df['day'].diff().dropna()
        if (day_diff < 0).any():
            print("WARNING: Days are not in chronological order!")
            print("This could cause data leakage when creating lagged features.")
        
        # Check for day column distribution similarity
        plt.figure(figsize=(10, 6))
        plt.hist(train_df['day'], bins=30, alpha=0.5, label='Train')
        if 'day' in test_df.columns:
            plt.hist(test_df['day'], bins=30, alpha=0.5, label='Test')
        plt.xlabel('Day')
        plt.ylabel('Count')
        plt.title('Distribution of Day Column')
        plt.legend()
        plt.savefig('/home/USER/Projects/RAIN/plots/day_distribution.png')
        print("Day distribution plot saved to '/home/USER/Projects/RAIN/plots/day_distribution.png'")
    
    # Check for feature distribution differences between train and test
    common_columns = [col for col in train_df.columns if col in test_df.columns and col != 'id' and col != 'rainfall']
    
    # Check distributions of numerical features
    for col in common_columns[:5]:  # Limit to first 5 to avoid too many plots
        if train_df[col].dtype in ['int64', 'float64']:
            plt.figure(figsize=(10, 6))
            plt.hist(train_df[col], bins=30, alpha=0.5, label='Train')
            plt.hist(test_df[col], bins=30, alpha=0.5, label='Test')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.title(f'Distribution of {col}')
            plt.legend()
            plt.savefig(f'/home/USER/Projects/RAIN/plots/{col}_distribution.png')
            print(f"Distribution plot for {col} saved")

def calibrate_predictions():
    """Calibrate model predictions to match expected distribution"""
    print("Testing prediction calibration...")
    
    # Load and preprocess data
    train_df, test_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # Apply feature engineering (best strategy)
    train_df, test_df = engineer_features(train_df, test_df, strategy='all_with_patterns')
    
    # Split data
    X = train_df.drop(['id', 'rainfall'], axis=1, errors='ignore')
    y = train_df['rainfall']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    import xgboost as xgb
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    val_preds = model.predict(X_val)
    test_preds = model.predict(test_df.drop(['id'], axis=1))
    
    # Check prediction distributions
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.hist(y_train, bins=30, alpha=0.5, label='Training Labels')
    plt.title('Training Labels')
    
    plt.subplot(1, 3, 2)
    plt.hist(val_preds, bins=30, alpha=0.5, label='Validation Predictions')
    plt.hist(y_val, bins=30, alpha=0.5, label='Validation Labels')
    plt.title('Validation: Predictions vs. Labels')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.hist(test_preds, bins=30, alpha=0.5, label='Test Predictions')
    plt.title('Test Predictions')
    
    plt.tight_layout()
    plt.savefig('/home/USER/Projects/RAIN/plots/prediction_distributions.png')
    print("Prediction distribution plot saved")
    
    # Calculate original score
    original_score = roc_auc_score(y_val, val_preds)
    
    # Apply calibration methods and compare
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LinearRegression
    
    # Isotonic calibration
    iso_calibrator = IsotonicRegression(out_of_bounds='clip')
    iso_calibrator.fit(val_preds.reshape(-1, 1), y_val)
    val_preds_iso = iso_calibrator.predict(val_preds.reshape(-1, 1))
    test_preds_iso = iso_calibrator.predict(test_preds.reshape(-1, 1))
    
    # Linear calibration
    linear_calibrator = LinearRegression()
    linear_calibrator.fit(val_preds.reshape(-1, 1), y_val)
    val_preds_linear = linear_calibrator.predict(val_preds.reshape(-1, 1))
    test_preds_linear = linear_calibrator.predict(test_preds.reshape(-1, 1))
    
    # Calculate calibrated scores
    iso_score = roc_auc_score(y_val, val_preds_iso)
    linear_score = roc_auc_score(y_val, val_preds_linear)
    
    print("\nCalibration results:")
    print(f"Original validation score: {original_score:.6f}")
    print(f"Isotonic calibration score: {iso_score:.6f}")
    print(f"Linear calibration score: {linear_score:.6f}")
    
    # Save calibrated test predictions
    best_calibration = 'isotonic' if iso_score > linear_score else 'linear'
    best_preds = test_preds_iso if best_calibration == 'isotonic' else test_preds_linear
    
    calibrated_submission = pd.DataFrame({
        'id': test_df['id'],
        'rainfall': best_preds
    })
    
    calibrated_submission.to_csv('/home/USER/Projects/RAIN/data/calibrated_submission.csv', index=False)
    print(f"Calibrated submission (using {best_calibration}) saved to '/home/USER/Projects/RAIN/data/calibrated_submission.csv'")

if __name__ == "__main__":
    analyze_validation_gap()
    check_for_data_leakage()
    perform_time_based_validation()
    calibrate_predictions()
