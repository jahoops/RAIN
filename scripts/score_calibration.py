import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
import pickle
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures

# Dictionary to store calibration models for different scoring methods
calibration_models = {}

def collect_score_pairs():
    """Collect pairs of validation scores and actual submission scores"""
    # Check for existing validation log
    if os.path.exists('/home/USER/Projects/RAIN/models/synthetic_validation_log.csv'):
        pairs_df = pd.read_csv('/home/USER/Projects/RAIN/models/synthetic_validation_log.csv')
        print(f"Loaded {len(pairs_df)} validation-submission score pairs from log")
        return pairs_df
    
    # If no log exists, try to find submission files with scores in their names
    score_pairs = []
    
    # Look for files like submission_all_score_0.882270.csv
    submission_files = glob.glob('/home/USER/Projects/RAIN/data/submission_*_score_*.csv')
    
    for file_path in submission_files:
        try:
            # Extract score and strategy from filename
            file_name = os.path.basename(file_path)
            
            # Try to extract validation score from filename
            score_match = re.search(r'score_([0-9.]+)', file_name)
            if not score_match:
                continue
                
            validation_score = float(score_match.group(1))
            
            # Try to extract feature strategy from filename
            strategy_match = re.search(r'submission_([^_]+)_', file_name)
            strategy = strategy_match.group(1) if strategy_match else 'unknown'
            
            # Look for this file in experiment_results.csv to find actual score
            if os.path.exists('/home/USER/Projects/RAIN/models/experiment_results.csv'):
                results_df = pd.read_csv('/home/USER/Projects/RAIN/models/experiment_results.csv')
                matching_row = results_df[results_df['CV Score'].round(6) == round(validation_score, 6)]
                
                if not matching_row.empty and 'Submission Score' in matching_row.columns:
                    actual_score = matching_row['Submission Score'].values[0]
                    
                    score_pairs.append({
                        'Submission_File': file_name,
                        'Strategy': strategy,
                        'Validation_Score': validation_score,
                        'Actual_Score': actual_score,
                        'Difference': validation_score - actual_score
                    })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if not score_pairs:
        print("No score pairs found. Please provide some validation-submission score pairs.")
        # Create a sample with the 0.03 difference mentioned
        score_pairs.append({
            'Submission_File': 'submission_all_with_patterns_catboost_score_0.925602.csv',
            'Strategy': 'all_with_patterns',
            'Validation_Score': 0.925602,
            'Actual_Score': 0.85518,  # ~0.07 lower than validation
            'Difference': 0.925602 - 0.85518
        })
    
    pairs_df = pd.DataFrame(score_pairs)
    print(f"Collected {len(pairs_df)} validation-submission score pairs")
    return pairs_df

def build_calibration_model(pairs_df=None):
    """Build a model to calibrate validation scores to expected submission scores"""
    if pairs_df is None:
        pairs_df = collect_score_pairs()
        
    if len(pairs_df) < 2:
        print("Not enough data points for regression. Using fixed adjustment.")
        mean_diff = pairs_df['Difference'].mean()
        print(f"Average difference between validation and actual scores: {mean_diff:.6f}")
        
        # Create simple calibration function
        def calibrate_score(validation_score):
            return validation_score - mean_diff
        
        return calibrate_score
    
    # With enough data, build a regression model
    X = pairs_df['Validation_Score'].values.reshape(-1, 1)
    y = pairs_df['Actual_Score'].values
    
    # Try both linear regression and ridge regression
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    lr_preds = lr_model.predict(X)
    lr_mae = mean_absolute_error(y, lr_preds)
    
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X, y)
    ridge_preds = ridge_model.predict(X)
    ridge_mae = mean_absolute_error(y, ridge_preds)
    
    # Use the better model
    if ridge_mae < lr_mae:
        print(f"Using Ridge regression (MAE: {ridge_mae:.6f})")
        model = ridge_model
    else:
        print(f"Using Linear regression (MAE: {lr_mae:.6f})")
        model = lr_model
    
    # Create calibration function from the model
    def calibrate_score(validation_score):
        return float(model.predict([[validation_score]])[0])
    
    # Visualize the calibration
    plt.figure(figsize=(10, 6))
    plt.scatter(pairs_df['Validation_Score'], pairs_df['Actual_Score'], alpha=0.7, label='Actual pairs')
    
    # Plot the calibration line
    x_range = np.linspace(
        max(0.5, pairs_df['Validation_Score'].min() - 0.05),
        min(1.0, pairs_df['Validation_Score'].max() + 0.05),
        100
    )
    y_range = [calibrate_score(x) for x in x_range]
    plt.plot(x_range, y_range, 'r-', label='Calibration model')
    
    # Add diagonal reference line
    plt.plot(x_range, x_range, 'g--', alpha=0.5, label='No adjustment')
    
    plt.xlabel('Validation Score')
    plt.ylabel('Expected Submission Score')
    plt.title('Score Calibration Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text showing the average difference
    mean_diff = pairs_df['Difference'].mean()
    plt.annotate(f"Average difference: {mean_diff:.4f}", 
                 xy=(0.05, 0.95), 
                 xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/home/USER/Projects/RAIN/plots/score_calibration.png')
    print("Saved calibration plot to '/home/USER/Projects/RAIN/plots/score_calibration.png'")
    
    return calibrate_score

def predict_submission_score(validation_score, model_name=None, feature_strategy=None):
    """Predict the expected submission score based on validation score"""
    # Build or load calibration model
    calibrate_score = build_calibration_model()
    
    # Apply calibration
    expected_submission_score = calibrate_score(validation_score)
    
    print(f"\nScore Calibration Results:")
    print(f"Input validation score: {validation_score:.6f}")
    print(f"Estimated submission score: {expected_submission_score:.6f}")
    print(f"Adjustment: {expected_submission_score - validation_score:.6f}")
    
    # Record this prediction
    if model_name and feature_strategy:
        prediction_log_path = '/home/USER/Projects/RAIN/models/score_predictions.csv'
        prediction = pd.DataFrame({
            'Timestamp': [pd.Timestamp.now()],
            'Model': [model_name],
            'Feature_Strategy': [feature_strategy],
            'Validation_Score': [validation_score],
            'Predicted_Submission': [expected_submission_score],
            'Adjustment': [expected_submission_score - validation_score]
        })
        
        if os.path.exists(prediction_log_path):
            log_df = pd.read_csv(prediction_log_path)
            log_df = pd.concat([log_df, prediction], ignore_index=True)
        else:
            log_df = prediction
            
        log_df.to_csv(prediction_log_path, index=False)
        
    return expected_submission_score

def train_calibration_model(synthetic_scores, actual_scores, method='default'):
    """
    Train a calibration model to map synthetic validation scores to actual submission scores
    
    Args:
        synthetic_scores: List of synthetic validation scores
        actual_scores: List of actual submission scores (ground truth)
        method: Scoring method used for synthetic scores
        
    Returns:
        model: Trained calibration model
    """
    print(f"Training calibration model for method '{method}' with {len(synthetic_scores)} data points")
    
    # Convert inputs to numpy arrays
    synthetic_scores_array = np.array(synthetic_scores)
    actual_scores_array = np.array(actual_scores)
    
    # Calculate the mean difference
    mean_diff = np.mean(actual_scores_array - synthetic_scores_array)
    print(f"Mean difference: {mean_diff:.6f}")
    
    if len(synthetic_scores) < 2:
        print("Not enough data points for regression. Using fixed adjustment.")
        return {'type': 'fixed_offset', 'offset': mean_diff}
    
    elif len(synthetic_scores) < 4:
        print("Using linear regression for calibration with limited data.")
        # For small datasets, use simple linear regression
        X = synthetic_scores_array.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, actual_scores_array)
        
        # Print the model details
        print(f"Linear model: y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}")
        
        # Calculate predictions and errors for each data point
        predictions = model.predict(X)
        errors = np.abs(predictions - actual_scores_array)
        mean_error = np.mean(errors)
        print(f"Mean absolute error: {mean_error:.6f}")
        
        return {
            'type': 'linear_regression',
            'model': model,
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'error': mean_error
        }
    
    else:
        print("Using polynomial regression for calibration.")
        # For more data points, use polynomial regression
        X = synthetic_scores_array.reshape(-1, 1)
        
        # Try both linear and polynomial regression
        poly = PolynomialFeatures(degree=min(len(synthetic_scores)-1, 3))
        X_poly = poly.fit_transform(X)
        
        # Use robust regression to handle potential outliers
        huber_model = HuberRegressor()
        huber_model.fit(X_poly, actual_scores_array)
        
        # Also try linear regression
        linear_model = LinearRegression()
        linear_model.fit(X.reshape(-1, 1), actual_scores_array)
        
        # Compare the models
        huber_preds = huber_model.predict(X_poly)
        linear_preds = linear_model.predict(X.reshape(-1, 1))
        
        huber_error = np.mean(np.abs(huber_preds - actual_scores_array))
        linear_error = np.mean(np.abs(linear_preds - actual_scores_array))
        
        print(f"Huber polynomial regression error: {huber_error:.6f}")
        print(f"Linear regression error: {linear_error:.6f}")
        
        # Use the model with lower error
        if linear_error <= huber_error:
            print("Using linear model (better performance)")
            return {
                'type': 'linear_regression',
                'model': linear_model,
                'coefficients': linear_model.coef_,
                'intercept': linear_model.intercept_,
                'error': linear_error
            }
        else:
            print("Using polynomial model (better performance)")
            return {
                'type': 'polynomial_regression',
                'model': huber_model,
                'poly': poly,
                'degree': min(len(synthetic_scores)-1, 3),
                'coefficients': huber_model.coef_,
                'intercept': huber_model.intercept_,
                'error': huber_error
            }

def load_score_pairs():
    """
    Load validation-submission score pairs from the validation log
    
    Returns:
        dict: Dictionary of scoring methods to (synthetic_scores, actual_scores) tuples
    """
    log_path = '/home/USER/Projects/RAIN/models/synthetic_validation_log.csv'
    score_pairs = {}
    
    if os.path.exists(log_path):
        try:
            validation_log = pd.read_csv(log_path)
            print(f"Loaded {len(validation_log)} entries from validation log")
            
            # Check if we have the right columns
            required_cols = ['Synthetic_Score', 'Actual_Score']
            if not all(col in validation_log.columns for col in required_cols):
                print("Warning: Validation log is missing required columns")
                print(f"Available columns: {validation_log.columns.tolist()}")
                if 'Scoring_Method' not in validation_log.columns:
                    # If we don't have scoring method, use missing columns but add default method
                    validation_log['Scoring_Method'] = 'default'
            else:
                # Group by scoring method
                if 'Scoring_Method' in validation_log.columns:
                    for method, group in validation_log.groupby('Scoring_Method'):
                        synthetic_scores = group['Synthetic_Score'].tolist()
                        actual_scores = group['Actual_Score'].tolist()
                        
                        if synthetic_scores and actual_scores:
                            score_pairs[method] = (synthetic_scores, actual_scores)
                            print(f"Found {len(synthetic_scores)} pairs for method '{method}'")
                else:
                    # No scoring method column - use everything as default
                    synthetic_scores = validation_log['Synthetic_Score'].tolist()
                    actual_scores = validation_log['Actual_Score'].tolist()
                    score_pairs['default'] = (synthetic_scores, actual_scores)
                    print(f"Found {len(synthetic_scores)} pairs without method specification")
        except Exception as e:
            print(f"Error loading validation log: {e}")
            import traceback
            traceback.print_exc()
    
    # Fall back to default if no method-specific data
    if not score_pairs and 'default' not in score_pairs:
        # Try to collect any available scores regardless of method
        if os.path.exists(log_path):
            try:
                validation_log = pd.read_csv(log_path)
                if 'Synthetic_Score' in validation_log.columns and 'Actual_Score' in validation_log.columns:
                    synthetic_scores = validation_log['Synthetic_Score'].tolist()
                    actual_scores = validation_log['Actual_Score'].tolist()
                    
                    if synthetic_scores and actual_scores:
                        score_pairs['default'] = (synthetic_scores, actual_scores)
                        print(f"Collected {len(synthetic_scores)} validation-submission score pairs for default calibration")
            except Exception as e:
                print(f"Error creating default score pairs: {e}")
    
    if not score_pairs:
        print("No score pairs found. Using default calibration.")
        # If we have absolutely no data, create a single default pair using typical pattern
        score_pairs['default'] = ([0.925602], [0.85518])  # Example from known submission
        print("Created fallback calibration using typical submission pattern")
    
    return score_pairs

def predict_submission_score(validation_score, method='default', force_calibration=False):
    """
    Predict actual submission score from synthetic validation score
    
    Args:
        validation_score: Synthetic validation score
        method: Scoring method used
        force_calibration: Force creation of a calibration model even with limited data
        
    Returns:
        float: Predicted submission score
    """
    global calibration_models
    
    # Load or train calibration model if needed
    if method not in calibration_models or force_calibration:
        score_pairs = load_score_pairs()
        
        if method in score_pairs:
            synthetic_scores, actual_scores = score_pairs[method]
            calibration_models[method] = train_calibration_model(synthetic_scores, actual_scores, method)
        elif 'default' in score_pairs:
            print(f"No data for method '{method}', using default calibration.")
            synthetic_scores, actual_scores = score_pairs['default']
            calibration_models[method] = train_calibration_model(synthetic_scores, actual_scores, 'default')
        else:
            print("No calibration data available. Using fixed adjustment based on typical patterns.")
            # Instead of identity, use a reasonable fixed adjustment based on observations
            # Typically, validation scores are 0.03-0.07 higher than actual LB scores
            calibration_models[method] = {'type': 'fixed_offset', 'offset': -0.05}
            print(f"Applied default adjustment of -0.05 (typical validation-to-submission gap)")
            
    # Make prediction using the calibration model
    model_info = calibration_models[method]
    model_type = model_info['type']
    
    if model_type == 'fixed_offset':
        adjusted = validation_score + model_info['offset']
        print(f"Applied fixed offset of {model_info['offset']:.4f}")
        return adjusted
    
    elif model_type == 'linear_regression':
        model = model_info['model']
        prediction = model.predict([[validation_score]])[0]
        print(f"Used linear model: {validation_score:.4f} → {prediction:.4f}")
        return prediction
    
    elif model_type == 'polynomial_regression':
        model = model_info['model']
        poly = model_info['poly']
        X_poly = poly.transform([[validation_score]])
        prediction = model.predict(X_poly)[0]
        print(f"Used polynomial model: {validation_score:.4f} → {prediction:.4f}")
        return prediction
    
    else:  # identity
        print("WARNING: Using identity function for calibration (no adjustment)")
        return validation_score

def save_calibration_models():
    """Save all calibration models to disk"""
    os.makedirs('/home/USER/Projects/RAIN/models', exist_ok=True)
    with open('/home/USER/Projects/RAIN/models/score_calibration_models.pkl', 'wb') as f:
        pickle.dump(calibration_models, f)
    print("Saved calibration models to disk")

def load_calibration_models():
    """Load calibration models from disk"""
    global calibration_models
    model_path = '/home/USER/Projects/RAIN/models/score_calibration_models.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            calibration_models = pickle.load(f)
        print(f"Loaded {len(calibration_models)} calibration models from disk")
    else:
        print("No saved calibration models found")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Calibrate validation scores to predict submission scores')
    parser.add_argument('--score', type=float, help='Validation score to calibrate')
    parser.add_argument('--model', type=str, help='Model name (for logging)')
    parser.add_argument('--strategy', type=str, help='Feature strategy (for logging)')
    
    args = parser.parse_args()
    
    if args.score:
        predict_submission_score(args.score, args.model, args.strategy)
    else:
        # Just build and visualize the calibration model
        calibrate_score = build_calibration_model()
        
        # Show a few example calibrations
        for score in [0.85, 0.90, 0.95]:
            print(f"Validation score {score:.2f} → Expected submission {calibrate_score(score):.4f} " +
                  f"(adjustment: {calibrate_score(score) - score:.4f})")
    
    # Load existing calibration models
    load_calibration_models()
    
    # Analyze existing score pairs
    score_pairs = load_score_pairs()
    print(f"Found {len(score_pairs)} scoring methods with calibration data")
    
    for method, (synthetic_scores, actual_scores) in score_pairs.items():
        print(f"\nMethod: {method}")
        print(f"  Data points: {len(synthetic_scores)}")
        
        # Calculate average difference
        avg_diff = np.mean(np.array(actual_scores) - np.array(synthetic_scores))
        print(f"  Average difference between validation and actual scores: {avg_diff:.6f}")
        
        # Test calibration on existing data
        print("\nScore Calibration Examples:")
        for synth, actual in zip(synthetic_scores, actual_scores):
            pred = predict_submission_score(synth, method=method)
            error = abs(pred - actual)
            print(f"  Synthetic: {synth:.6f}, Actual: {actual:.6f}, Predicted: {pred:.6f}, Error: {error:.6f}")
    
    # Save updated models
    save_calibration_models()
