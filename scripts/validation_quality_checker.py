import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, roc_auc_score
import os
from score_calibration import predict_submission_score

def analyze_validation_methods():
    """Analyze how well different validation methods predict submission scores"""
    
    # Load experiment results if available
    try:
        results_df = pd.read_csv('models/experiment_results.csv')
        print("Found experiment results with the following columns:")
        print(results_df.columns.tolist())
        
        # Check if we have submission scores recorded
        if 'Submission Score' in results_df.columns:
            # Analyze correlation between different validation methods and actual submission score
            correlation_cv = results_df['CV Score'].corr(results_df['Submission Score'])
            correlation_synthetic = results_df['Synthetic Score'].corr(results_df['Submission Score'])
            
            print(f"Correlation between CV scores and submission scores: {correlation_cv:.4f}")
            print(f"Correlation between synthetic validation scores and submission scores: {correlation_synthetic:.4f}")
            
            # Calculate mean absolute errors
            mae_cv = np.mean(np.abs(results_df['CV Score'] - results_df['Submission Score']))
            mae_synthetic = np.mean(np.abs(results_df['Synthetic Score'] - results_df['Submission Score']))
            
            print(f"Mean absolute error for CV prediction: {mae_cv:.4f}")
            print(f"Mean absolute error for synthetic validation prediction: {mae_synthetic:.4f}")
            
            # Visualize the relationships
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.scatter(results_df['CV Score'], results_df['Submission Score'], alpha=0.7)
            plt.plot([min(results_df['CV Score']), max(results_df['CV Score'])], 
                     [min(results_df['CV Score']), max(results_df['CV Score'])], 'r--')
            plt.xlabel('CV Score')
            plt.ylabel('Submission Score')
            plt.title(f'CV vs Submission (r={correlation_cv:.4f})')
            
            plt.subplot(1, 2, 2)
            plt.scatter(results_df['Synthetic Score'], results_df['Submission Score'], alpha=0.7)
            plt.plot([min(results_df['Synthetic Score']), max(results_df['Synthetic Score'])], 
                    [min(results_df['Synthetic Score']), max(results_df['Synthetic Score'])], 'r--')
            plt.xlabel('Synthetic Validation Score')
            plt.ylabel('Submission Score')
            plt.title(f'Synthetic vs Submission (r={correlation_synthetic:.4f})')
            
            plt.tight_layout()
            plt.savefig('plots/validation_methods_comparison.png')
            print("Saved comparison plot to 'plots/validation_methods_comparison.png'")
            
            # Create a calibration model to predict submission scores
            from sklearn.linear_model import LinearRegression
            
            # Use both CV and synthetic scores to predict submission score
            X_calib = results_df[['CV Score', 'Synthetic Score', 'Gap']]
            y_calib = results_df['Submission Score']
            
            model = LinearRegression()
            model.fit(X_calib, y_calib)
            
            print("\nCalibration model coefficients:")
            print(f"Intercept: {model.intercept_:.4f}")
            for name, coef in zip(['CV Score', 'Synthetic Score', 'Gap'], model.coef_):
                print(f"{name}: {coef:.4f}")
            
            # Calculate calibrated predictions
            results_df['Calibrated Prediction'] = model.predict(X_calib)
            
            # Calculate new error
            mae_calibrated = np.mean(np.abs(results_df['Calibrated Prediction'] - results_df['Submission Score']))
            print(f"Mean absolute error after calibration: {mae_calibrated:.4f}")
            
            # Save the calibration model
            import pickle
            os.makedirs('models', exist_ok=True)
            pickle.dump(model, open('models/score_calibration_model.pkl', 'wb'))
            print("Saved calibration model to 'models/score_calibration_model.pkl'")
            
            return model, mae_calibrated < mae_cv
        else:
            print("No submission scores found in experiment results.")
            print("To use this feature, add actual competition scores to experiment_results.csv")
            print("Add a column 'Submission Score' with the final public leaderboard scores")
            return None, False
    except Exception as e:
        print(f"Could not analyze validation methods: {e}")
        print("Run experiments with different strategies and record submission scores first")
        return None, False

def recommend_synthetic_data_improvements():
    """Analyze synthetic data quality and recommend improvements"""
    # Try to load synthetic and training data
    try:
        methods = ['bootstrap', 'parametric', 'copula', 'perturbation', 'weather']
        train_df = pd.read_csv('data/train.csv')
        
        findings = []
        
        for method in methods:
            try:
                synth_path = f'data/synthetic_{method}.csv'
                if os.path.exists(synth_path):
                    synth_df = pd.read_csv(synth_path)
                    
                    # Compare distribution statistics
                    common_cols = list(set(train_df.columns) & set(synth_df.columns))
                    common_cols = [c for c in common_cols if train_df[c].dtype != 'object']
                    
                    if 'rainfall' in common_cols:
                        # Compare rainfall distributions
                        ks_stat = np.abs(np.sort(train_df['rainfall']) - 
                                        np.sort(synth_df['rainfall'].sample(
                                            n=min(len(train_df), len(synth_df)),
                                            replace=True))).max()
                        
                        findings.append({
                            'Method': method,
                            'KS_Distance': ks_stat,
                            'Mean_Diff': abs(train_df['rainfall'].mean() - synth_df['rainfall'].mean()),
                            'Std_Diff': abs(train_df['rainfall'].std() - synth_df['rainfall'].std()),
                            'Min_Diff': abs(train_df['rainfall'].min() - synth_df['rainfall'].min()),
                            'Max_Diff': abs(train_df['rainfall'].max() - synth_df['rainfall'].max())
                        })
            except Exception as e:
                print(f"Error analyzing {method} synthetic data: {e}")
        
        if findings:
            findings_df = pd.DataFrame(findings)
            print("\nSynthetic data quality comparison:")
            print(findings_df)
            
            # Find best method
            best_method = findings_df.loc[findings_df['KS_Distance'].idxmin()]['Method']
            print(f"\nBased on distribution similarity, the best method is: {best_method}")
            
            # Make recommendations
            print("\nRecommendations for synthetic data improvement:")
            if findings_df['KS_Distance'].min() > 0.1:
                print("- All methods have significant distribution differences")
                print("- Try hybrid approaches combining multiple methods")
                print("- Focus on preserving rainfall pattern correlations with other features")
            else:
                print(f"- Use {best_method} as the primary synthetic data generation method")
                print("- Consider ensemble validation using multiple synthetic sets")
            
            return findings_df
    except Exception as e:
        print(f"Could not analyze synthetic data: {e}")
        return None

def validate_with_real_submission(submission_file, actual_score):
    """
    Validate synthetic test approach using a real submission file with known score
    
    Args:
        submission_file: Path to the submission file
        actual_score: The actual score received for this submission
    """
    print(f"Validating synthetic method using submission: {submission_file}")
    print(f"Known actual score: {actual_score}")
    
    try:
        # Load the submission file
        submission_df = pd.read_csv(submission_file)
        print(f"Submission file contains {len(submission_df)} predictions")
        
        # Load synthetic test data
        synthetic_test_path = '/home/USER/Projects/RAIN/data/synthetic_test_set.csv'
        
        if not os.path.exists(synthetic_test_path):
            print("No synthetic test set found. Creating one...")
            from validation_with_synthetic import create_synthetic_test_set
            synthetic_test_df = create_synthetic_test_set(n_samples=2000, method='ensemble')
        else:
            synthetic_test_df = pd.read_csv(synthetic_test_path)
            print(f"Loaded synthetic test set with {len(synthetic_test_df)} samples")
        
        # Calculate synthetic validation score
        if 'rainfall' in synthetic_test_df.columns and len(submission_df) == len(synthetic_test_df):
            # If sizes match, assume the submission corresponds to the synthetic data
            synthetic_score = roc_auc_score(
                synthetic_test_df['rainfall'], 
                submission_df['rainfall']
            )
            print(f"Synthetic validation score: {synthetic_score:.6f}")
            print(f"Actual submission score: {actual_score}")
            print(f"Absolute difference: {abs(synthetic_score - actual_score):.6f}")
            
            # Add calibration
            calibrated_score = predict_submission_score(synthetic_score)
            print(f"Calibrated synthetic score: {calibrated_score:.6f}")
            print(f"Calibrated difference: {abs(calibrated_score - actual_score):.6f}")
            
            # Calculate error percentage
            error_pct = abs(synthetic_score - actual_score) / actual_score * 100
            print(f"Error percentage: {error_pct:.2f}%")
            
            # Create or update a log of validation results
            log_entry = pd.DataFrame({
                'Submission_File': [os.path.basename(submission_file)],
                'Synthetic_Score': [synthetic_score],
                'Calibrated_Score': [calibrated_score],
                'Actual_Score': [actual_score],
                'Raw_Difference': [synthetic_score - actual_score],
                'Calibrated_Difference': [calibrated_score - actual_score],
                'Error_Pct': [error_pct],
                'Timestamp': [pd.Timestamp.now()]
            })
            
            log_path = '/home/USER/Projects/RAIN/models/synthetic_validation_log.csv'
            if os.path.exists(log_path):
                validation_log = pd.read_csv(log_path)
                validation_log = pd.concat([validation_log, log_entry], ignore_index=True)
            else:
                validation_log = log_entry
                
            validation_log.to_csv(log_path, index=False)
            print(f"Updated validation log at {log_path}")
            
            return synthetic_score, actual_score
        else:
            # If sizes don't match, we need a more sophisticated approach
            print("Submission and synthetic test set sizes don't match or missing 'rainfall' column.")
            print("Trying alternative validation approach...")
            
            # Try to match features between submission and synthetic data
            if 'id' in submission_df.columns and 'id' in synthetic_test_df.columns:
                # If we have IDs, we can try to match the predictions
                common_ids = set(submission_df['id']).intersection(set(synthetic_test_df['id']))
                if common_ids:
                    print(f"Found {len(common_ids)} common IDs for validation")
                    
                    # Filter to common IDs
                    submission_subset = submission_df[submission_df['id'].isin(common_ids)]
                    synthetic_subset = synthetic_test_df[synthetic_test_df['id'].isin(common_ids)]
                    
                    # Sort both by ID to ensure alignment
                    submission_subset = submission_subset.sort_values('id')
                    synthetic_subset = synthetic_subset.sort_values('id')
                    
                    # Check if we need to convert rainfall values for ROC AUC calculation
                    # ROC AUC requires binary labels for ground truth (or at least 2 classes)
                    if 'rainfall' in synthetic_subset.columns:
                        # Try multiple scoring approaches to find the best one
                        scores = {}
                        
                        # 1. Binary classification approach
                        try:
                            # Check distribution of values in synthetic dataset
                            unique_values = np.unique(synthetic_subset['rainfall'])
                            if len(unique_values) <= 2:
                                # Already binary - can use directly
                                y_true = synthetic_subset['rainfall'].values
                                binary_score = roc_auc_score(y_true, submission_subset['rainfall'].values)
                                scores['binary_original'] = binary_score
                                print(f"Binary score (original): {binary_score:.6f}")
                            else:
                                # Convert to binary for AUC calculation using different thresholds
                                # Use more reasonable threshold values
                                thresholds = [
                                    float(synthetic_subset['rainfall'].mean()),
                                    float(synthetic_subset['rainfall'].median()),
                                    0.5,  # Common threshold for probability
                                    1.0   # Common threshold for rain/no rain
                                ]
                                
                                for i, threshold in enumerate(thresholds):
                                    threshold_name = ["mean", "median", "0.5", "1.0"][i]
                                    try:
                                        # Ensure threshold is a reasonable value
                                        if not np.isfinite(threshold) or threshold > 1e10:
                                            print(f"Skipping invalid threshold value: {threshold}")
                                            continue
                                            
                                        y_true = (synthetic_subset['rainfall'] > threshold).astype(int).values
                                        # Check if we have variation in the binary values
                                        if len(np.unique(y_true)) < 2:
                                            print(f"Threshold {threshold} ({threshold_name}) creates all {y_true[0]} values, skipping")
                                            continue
                                            
                                        binary_score = roc_auc_score(y_true, submission_subset['rainfall'].values)
                                        scores[f'binary_threshold_{threshold_name}'] = binary_score
                                        print(f"Binary score (threshold {threshold} [{threshold_name}]): {binary_score:.6f}")
                                    except Exception as e:
                                        print(f"Error calculating binary score with threshold {threshold}: {e}")
                        except Exception as e:
                            print(f"Error in binary scoring approach: {e}")
                        
                        # 2. Regression approach
                        try:
                            # Calculate regression metrics
                            mse = mean_squared_error(synthetic_subset['rainfall'].values, 
                                                    submission_subset['rainfall'].values)
                            scores['mse'] = mse
                            print(f"MSE: {mse:.6f}")
                            
                            # Convert MSE to a score-like metric (higher is better)
                            mse_score = 1 / (1 + mse)
                            scores['mse_score'] = mse_score
                            print(f"MSE Score: {mse_score:.6f}")
                            
                            # Calculate correlation
                            corr = np.corrcoef(synthetic_subset['rainfall'].values, 
                                              submission_subset['rainfall'].values)[0, 1]
                            scores['correlation'] = abs(corr)
                            print(f"Correlation: {corr:.6f}")
                        except Exception as e:
                            print(f"Error in regression scoring approach: {e}")
                        
                        # 3. Rank-based approach
                        try:
                            from scipy.stats import spearmanr
                            rank_corr, _ = spearmanr(synthetic_subset['rainfall'].values, 
                                                   submission_subset['rainfall'].values)
                            scores['rank_correlation'] = abs(rank_corr)
                            print(f"Rank Correlation: {rank_corr:.6f}")
                        except Exception as e:
                            print(f"Error in rank correlation approach: {e}")
                        
                        # Determine the most promising approach based on correlation with known scores
                        if scores:
                            # Default to most reasonable score, prioritizing median threshold
                            if 'binary_threshold_median' in scores:
                                synthetic_score = scores['binary_threshold_median']
                                scoring_method = 'binary_classification_median'
                            elif 'binary_threshold_mean' in scores:
                                synthetic_score = scores['binary_threshold_mean']
                                scoring_method = 'binary_classification_mean'
                            elif 'binary_threshold_1.0' in scores:
                                synthetic_score = scores['binary_threshold_1.0']
                                scoring_method = 'binary_classification_1.0'
                            elif 'binary_original' in scores:
                                synthetic_score = scores['binary_original']
                                scoring_method = 'binary_original'
                            elif 'mse_score' in scores:
                                synthetic_score = scores['mse_score']
                                scoring_method = 'mse_score'
                            elif 'correlation' in scores:
                                synthetic_score = scores['correlation']
                                scoring_method = 'correlation'
                            else:
                                # Use the first available score
                                scoring_method = list(scores.keys())[0]
                                synthetic_score = scores[scoring_method]
                            
                            print(f"Selected validation method: {scoring_method}")
                            print(f"Synthetic validation score: {synthetic_score:.6f}")
                            print(f"Actual submission score: {actual_score}")
                            print(f"Absolute difference: {abs(synthetic_score - actual_score):.6f}")
                            
                            # Add calibration - force creation of a reasonable calibration
                            try:
                                # Create initial calibration if none exists
                                # First save this pair to ensure we have data to calibrate with
                                log_entry = pd.DataFrame({
                                    'Submission_File': [os.path.basename(submission_file)],
                                    'Scoring_Method': [scoring_method],
                                    'Synthetic_Score': [synthetic_score],
                                    'Actual_Score': [actual_score],
                                    'Timestamp': [pd.Timestamp.now()]
                                })
                                
                                log_path = '/home/USER/Projects/RAIN/models/synthetic_validation_log.csv'
                                if os.path.exists(log_path):
                                    validation_log = pd.read_csv(log_path)
                                    # Check if this exact file is already in the log
                                    if not validation_log[validation_log['Submission_File'] == os.path.basename(submission_file)].empty:
                                        print("This submission is already in the log. Using existing data.")
                                    else:
                                        validation_log = pd.concat([validation_log, log_entry], ignore_index=True)
                                        validation_log.to_csv(log_path, index=False)
                                        print(f"Added to validation log at {log_path}")
                                else:
                                    os.makedirs(os.path.dirname(log_path), exist_ok=True)
                                    log_entry.to_csv(log_path, index=False)
                                    print(f"Created validation log at {log_path}")
                                
                                # Now request calibration
                                from score_calibration import predict_submission_score
                                calibrated_score = predict_submission_score(synthetic_score, method=scoring_method, force_calibration=True)
                                print(f"Calibrated synthetic score: {calibrated_score:.6f}")
                                print(f"Calibrated difference: {abs(calibrated_score - actual_score):.6f}")
                                
                                # Calculate error percentage
                                error_pct = abs(calibrated_score - actual_score) / actual_score * 100
                                print(f"Error percentage: {error_pct:.2f}%")
                                
                                # Save calibrated results
                                calib_log_entry = pd.DataFrame({
                                    'Submission_File': [os.path.basename(submission_file)],
                                    'Scoring_Method': [scoring_method],
                                    'Synthetic_Score': [synthetic_score],
                                    'Calibrated_Score': [calibrated_score],
                                    'Actual_Score': [actual_score],
                                    'Raw_Difference': [synthetic_score - actual_score],
                                    'Calibrated_Difference': [calibrated_score - actual_score],
                                    'Error_Pct': [error_pct],
                                    'Timestamp': [pd.Timestamp.now()]
                                })
                                
                                calib_log_path = '/home/USER/Projects/RAIN/models/calibration_results.csv'
                                if os.path.exists(calib_log_path):
                                    calib_log = pd.read_csv(calib_log_path)
                                    calib_log = pd.concat([calib_log, calib_log_entry], ignore_index=True)
                                else:
                                    calib_log = calib_log_entry
                                
                                calib_log.to_csv(calib_log_path, index=False)
                                print(f"Saved calibration results to {calib_log_path}")
                                
                            except Exception as e:
                                print(f"Could not apply calibration: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            return synthetic_score, actual_score
                        else:
                            print("No valid scoring methods found for this data")
                    else:
                        print("No 'rainfall' column found in synthetic test set")
                else:
                    print("No common IDs found between submission and synthetic data.")
            
            print("Cannot validate submission against synthetic data with current approach.")
            return None, actual_score
    
    except Exception as e:
        print(f"Error validating submission: {e}")
        import traceback
        traceback.print_exc()  # Print full stack trace for debugging
        return None, actual_score

if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    os.makedirs('/home/USER/Projects/RAIN/plots', exist_ok=True)
    os.makedirs('/home/USER/Projects/RAIN/models', exist_ok=True)
    
    print("Validation Quality Checker\n")
    
    # New option to validate using real submission
    import argparse
    parser = argparse.ArgumentParser(description='Validate synthetic testing approach')
    parser.add_argument('--submission', type=str, help='Path to submission file to validate')
    parser.add_argument('--actual-score', type=float, help='Actual score for the submission')
    args = parser.parse_args()
    
    if args.submission and args.actual_score:
        print(f"Validating using submission file: {args.submission}")
        validate_with_real_submission(args.submission, args.actual_score)
    else:
        # Run the original analysis
        print("Analyzing validation methods effectiveness...\n")
        model, is_better = analyze_validation_methods()
        
        print("\nAnalyzing synthetic data quality and making recommendations...")
        findings = recommend_synthetic_data_improvements()
        
        # Also check if we have any submission files with scores in their names
        import glob
        submission_files = glob.glob('/home/USER/Projects/RAIN/data/submission_*_score_*.csv')
        
        if submission_files:
            print("\nFound submission files with scores in their names:")
            for submission_file in submission_files:
                try:
                    # Try to extract score from filename
                    import re
                    # Updated regex pattern to handle the format correctly
                    score_match = re.search(r'score_([0-9]+\.[0-9]+)', submission_file)
                    if score_match:
                        score = float(score_match.group(1))
                        print(f"Validating {os.path.basename(submission_file)} with score {score}")
                        validate_with_real_submission(submission_file, score)
                except Exception as e:
                    print(f"Error processing {submission_file}: {e}")

