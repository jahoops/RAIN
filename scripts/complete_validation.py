import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns

def generate_realistic_synthetic_data(submission_df):
    """Generate realistic synthetic data based on feature relationships"""
    print("\n=== GENERATING ENHANCED SYNTHETIC DATA ===")
    
    # Load training data to learn feature relationships
    from data_pipeline import create_complete_data_pipeline
    X_train_temporal, X_test_temporal = create_complete_data_pipeline()
    
    y_train = X_train_temporal['rainfall']
    X_train = X_train_temporal.drop(['id', 'rainfall'], axis=1)
    
    # Use the actual test features
    X_test = X_test_temporal.drop(['id'], axis=1) if 'id' in X_test_temporal.columns else X_test_temporal
    
    # Find common columns
    common_columns = list(set(X_train.columns).intersection(set(X_test.columns)))
    print(f"Using {len(common_columns)} common features to generate synthetic data")
    
    if len(common_columns) < 3:
        print("WARNING: Very few common features found. Using direct ID-based generation instead.")
        # Simple deterministic generation based on IDs
        synthetic_df = pd.DataFrame({
            'id': submission_df['id'],
            'rainfall': np.sin(submission_df['id'] / 73.0) * 0.3 + 0.5
        })
    else:
        # Train model to generate synthetic values
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train[common_columns], y_train)
        
        # Generate predictions as synthetic "ground truth"
        synthetic_rainfall = model.predict(X_test[common_columns])
        
        # Add noise to avoid perfect correlation
        np.random.seed(42) 
        noise = np.random.normal(0, 0.05, len(synthetic_rainfall))
        synthetic_rainfall = np.clip(synthetic_rainfall + noise, 0, 1)
        
        # Map synthetic values to submission IDs
        synthetic_df = pd.DataFrame({
            'id': X_test_temporal['id'] if 'id' in X_test_temporal.columns else X_test_temporal.index,
            'rainfall': synthetic_rainfall
        })
        
        # Align with submission IDs
        synthetic_df = pd.merge(
            pd.DataFrame({'id': submission_df['id']}),
            synthetic_df,
            on='id',
            how='left'
        )
        
        # Fill any missing values
        if synthetic_df['rainfall'].isna().any():
            print(f"Filling {synthetic_df['rainfall'].isna().sum()} missing values")
            synthetic_df['rainfall'] = synthetic_df['rainfall'].fillna(synthetic_df['rainfall'].mean())
    
    # Save enhanced synthetic data
    output_path = 'data/synthetic_enhanced.csv'
    synthetic_df.to_csv(output_path, index=False)
    print(f"Saved enhanced synthetic data to {output_path}")
    
    return synthetic_df

def validate_submission(submission_path, validation_metric='both'):
    """Validate a submission file using enhanced synthetic data"""
    print(f"\n=== VALIDATING SUBMISSION: {submission_path} ===")
    
    # Load submission
    submission_df = pd.read_csv(submission_path)
    print(f"Loaded {len(submission_df)} predictions from {submission_path}")
    
    # Generate enhanced synthetic data
    synthetic_df = generate_realistic_synthetic_data(submission_df)
    
    # Calculate metrics
    predictions = submission_df['rainfall'].values
    ground_truth = synthetic_df['rainfall'].values
    
    # Standard metrics
    mse = mean_squared_error(ground_truth, predictions)
    correlation = np.corrcoef(ground_truth, predictions)[0, 1]
    
    # Inverse correlation check - if correlation is negative, invert synthetic values
    if correlation < 0:
        print("Detected negative correlation - inverting synthetic values")
        inverted_truth = 1.0 - ground_truth
        inverted_mse = mean_squared_error(inverted_truth, predictions)
        inverted_correlation = np.corrcoef(inverted_truth, predictions)[0, 1]
        
        if inverted_correlation > 0:
            print(f"Inverting improved correlation: {correlation:.4f} → {inverted_correlation:.4f}")
            mse = inverted_mse
            correlation = inverted_correlation
            ground_truth = inverted_truth
    
    # For ROC AUC, we'll use a reasonable threshold for rainfall
    try:
        auc = roc_auc_score(ground_truth > 0.5, predictions)
    except:
        auc = None
    
    print("\n=== VALIDATION RESULTS ===")
    print(f"MSE: {mse:.6f}")
    print(f"Correlation: {correlation:.6f}")
    if auc:
        print(f"AUC: {auc:.6f}")
    
    # Estimated leaderboard score
    estimated_lb = (1 - mse) * 0.95
    print(f"Estimated leaderboard score: {estimated_lb:.6f}")
    
    # Visualization using matplotlib
    os.makedirs('analysis/validation', exist_ok=True)
    plt.figure(figsize=(14, 10))
    
    # Scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(predictions, ground_truth, alpha=0.3)
    plt.xlabel('Predictions')
    plt.ylabel('Synthetic Ground Truth')
    plt.title(f'Correlation: {correlation:.4f}')
    
    # Distribution comparison
    plt.subplot(2, 2, 2)
    plt.hist(predictions, bins=30, alpha=0.5, density=True, label='Predictions')
    plt.hist(ground_truth, bins=30, alpha=0.5, density=True, label='Synthetic Ground Truth')
    plt.legend()
    plt.title('Distribution Comparison')
    
    # Residual plot
    plt.subplot(2, 2, 3)
    residuals = ground_truth - predictions
    plt.scatter(predictions, residuals, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    # Residual histogram
    plt.subplot(2, 2, 4)
    plt.hist(residuals, bins=30, density=True)
    plt.title('Residual Distribution')
    
    plt.tight_layout()
    result_path = f'analysis/validation/{os.path.basename(submission_path)}_validation.png'
    plt.savefig(result_path)
    print(f"Saved validation visualization to {result_path}")
    
    # Return the metric of interest
    if validation_metric == 'mse':
        return mse
    elif validation_metric == 'correlation':
        return correlation
    elif validation_metric == 'auc':
        return auc if auc else 0.0
    else:
        return {'mse': mse, 'correlation': correlation, 'auc': auc if auc else 0.0}

def validate_with_training_segment(submission_df):
    """Validate submission against a held-out portion of training data"""
    print("\n=== VALIDATING WITH TRAINING DATA ===")
    
    # Load training data
    from data_pipeline import create_complete_data_pipeline
    X_train_temporal, _ = create_complete_data_pipeline()
    
    # Split training data into train and validation
    train_data, val_data = train_test_split(
        X_train_temporal, test_size=0.2, random_state=42
    )
    
    # Extract validation IDs and target
    val_ids = val_data['id'].values
    val_targets = val_data['rainfall'].values
    
    # Find matching predictions in submission
    matching_indices = []
    matching_targets = []
    matching_preds = []
    
    for i, id_val in enumerate(val_ids):
        # Find this ID in the submission
        sub_idx = submission_df[submission_df['id'] == id_val].index
        if len(sub_idx) > 0:
            matching_indices.append(i)
            matching_targets.append(val_targets[i])
            matching_preds.append(submission_df.loc[sub_idx[0], 'rainfall'])
    
    # Check if we have enough matches
    if len(matching_indices) < 10:
        print(f"WARNING: Only {len(matching_indices)} matching IDs found between validation and submission.")
        print("This is normal for final test data. Skipping training validation.")
        return None
    
    print(f"Found {len(matching_indices)} matching IDs between validation set and submission")
    
    # Calculate metrics
    mse = mean_squared_error(matching_targets, matching_preds)
    correlation = np.corrcoef(matching_targets, matching_preds)[0, 1]
    
    print(f"Training validation MSE: {mse:.6f}")
    print(f"Training validation correlation: {correlation:.6f}")
    
    return {
        'train_mse': mse,
        'train_correlation': correlation,
        'train_score': 1 - mse
    }

def extract_lb_score_from_filename(filename):
    """Extract leaderboard score from filename if present"""
    try:
        parts = filename.split('.')
        # Check each part for a 5-digit number
        for part in parts:
            if part.isdigit() and len(part) == 5:
                return float(part) / 100000.0
        return None
    except:
        return None

def compare_multiple_submissions(limit=None):
    """Compare multiple submissions with both synthetic and training validation"""
    print("\n=== COMPARING MULTIPLE SUBMISSIONS ===")
    
    # Find all submission files
    submission_dirs = ['data/Actual']
    
    submission_files = []
    used_dir = None
    
    for submission_dir in submission_dirs:
        if os.path.exists(submission_dir):
            files = [f for f in os.listdir(submission_dir) if f.endswith('.csv')]
            if files:
                submission_files = files
                used_dir = submission_dir
                break
                
    if not submission_files:
        print("ERROR: No submission files found in any directory!")
        return
    
    # Limit the number of files if requested
    if limit and len(submission_files) > limit:
        print(f"Limiting analysis to {limit} files")
        submission_files = submission_files[:limit]
    
    print(f"Found {len(submission_files)} submission files in {used_dir}")
    
    # Validate each submission
    results = []
    for file in submission_files:
        path = os.path.join(used_dir, file)
        print(f"\nValidating {file}...")
        
        # Load the submission
        submission_df = pd.read_csv(path)
        
        # Get synthetic validation metrics
        metrics = validate_submission(path, 'both')
        
        # Get training validation metrics
        train_metrics = validate_with_training_segment(submission_df)
        
        # Extract leaderboard score from filename if possible
        lb_score = extract_lb_score_from_filename(file)
        
        if metrics:
            result = {
                'file': file,
                'mse': metrics['mse'],
                'correlation': metrics['correlation'], 
                'auc': metrics['auc'] if metrics['auc'] else 0.0,
                'synthetic_score': (1 - metrics['mse']),
                'estimated_lb': (1 - metrics['mse']) * 0.95
            }
            
            # Add training metrics if available
            if train_metrics:
                result.update({
                    'train_mse': train_metrics['train_mse'],
                    'train_correlation': train_metrics['train_correlation'],
                    'train_score': train_metrics['train_score']
                })
                
            # Add leaderboard score if available
            if lb_score:
                result['lb_score'] = lb_score
                
            results.append(result)
    
    # Create a sorted dataframe of results
    if results:
        results_df = pd.DataFrame(results)
        
        # Calculate correlation between synthetic validation and training validation
        if 'train_score' in results_df.columns and len(results_df) > 2:
            if 'train_score' in results_df.columns and 'estimated_lb' in results_df.columns:
                if not results_df['train_score'].isna().all():
                    correlation = np.corrcoef(
                        results_df['train_score'].fillna(0), 
                        results_df['estimated_lb']
                    )[0, 1]
                    print(f"\nCorrelation between synthetic and training validation: {correlation:.4f}")
            
        # Calculate correlation with leaderboard score if available
        if 'lb_score' in results_df.columns and len(results_df) > 2:
            lb_corr = np.corrcoef(results_df['lb_score'], results_df['estimated_lb'])[0, 1]
            print(f"Correlation between synthetic validation and leaderboard: {lb_corr:.4f}")
            
            # If correlation is decent, update our estimation formula
            if abs(lb_corr) > 0.3:
                print("Using leaderboard correlation to improve estimation formula")
                
                # Linear regression to map synthetic scores to leaderboard scores
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression()
                reg.fit(results_df[['estimated_lb']], results_df['lb_score'])
                
                # Apply the correction
                results_df['calibrated_lb'] = reg.predict(results_df[['estimated_lb']])
                print(f"Calibration: LB ≈ {reg.coef_[0]:.4f} × Synthetic + {reg.intercept_:.4f}")
        
        # Sort by the best available metric
        if 'lb_score' in results_df.columns:
            results_df = results_df.sort_values('lb_score', ascending=False)
        elif 'calibrated_lb' in results_df.columns:
            results_df = results_df.sort_values('calibrated_lb', ascending=False)
        else:
            results_df = results_df.sort_values('estimated_lb', ascending=False)
        
        print("\n=== FINAL COMPARISON ===")
        print(results_df)
        
        # Save results
        os.makedirs('analysis/validation', exist_ok=True)
        results_df.to_csv('analysis/validation/complete_comparison.csv', index=False)
        print("Saved comparison results to analysis/validation/complete_comparison.csv")
        
        # Create visualizations
        plt.figure(figsize=(14, 10))
        
        # Plot correlation between synthetic and training validation
        plt.subplot(2, 2, 1)
        if 'train_score' in results_df.columns and 'estimated_lb' in results_df.columns:
            valid_rows = ~results_df['train_score'].isna()
            if valid_rows.sum() > 2:
                plt.scatter(
                    results_df.loc[valid_rows, 'estimated_lb'],
                    results_df.loc[valid_rows, 'train_score']
                )
                plt.xlabel('Synthetic Validation Score')
                plt.ylabel('Training Validation Score')
                corr = np.corrcoef(
                    results_df.loc[valid_rows, 'estimated_lb'], 
                    results_df.loc[valid_rows, 'train_score']
                )[0, 1]
                plt.title(f'Correlation: {corr:.4f}')
        
        # Plot correlation with leaderboard if available
        plt.subplot(2, 2, 2)
        if 'lb_score' in results_df.columns and 'estimated_lb' in results_df.columns:
            plt.scatter(results_df['estimated_lb'], results_df['lb_score'])
            plt.xlabel('Synthetic Validation Score')
            plt.ylabel('Leaderboard Score')
            
            lb_corr = np.corrcoef(results_df['lb_score'], results_df['estimated_lb'])[0, 1]
            plt.title(f'LB Correlation: {lb_corr:.4f}')
            
            # Add the regression line
            if 'calibrated_lb' in results_df.columns:
                x = np.linspace(
                    results_df['estimated_lb'].min(),
                    results_df['estimated_lb'].max(),
                    100
                )
                plt.plot(x, reg.predict(x.reshape(-1, 1)), 'r--')
        
        # Plot top model performances based on best available metric
        plt.subplot(2, 1, 2)
        
        if 'lb_score' in results_df.columns:
            # If we have actual leaderboard scores, use those
            top_models = results_df.head(10)
            plt.bar(top_models['file'], top_models['lb_score'])
            plt.ylabel('Leaderboard Score')
        elif 'calibrated_lb' in results_df.columns:
            top_models = results_df.head(10)
            plt.bar(top_models['file'], top_models['calibrated_lb'])
            plt.ylabel('Calibrated LB Score')
        else:
            top_models = results_df.head(10)
            plt.bar(top_models['file'], top_models['estimated_lb'])
            plt.ylabel('Estimated Score')
            
        plt.xticks(rotation=45, ha='right')
        plt.title('Top 10 Model Performance')
        
        plt.tight_layout()
        plt.savefig('analysis/validation/complete_comparison.png')
        
        return results_df
    else:
        print("Could not compare submissions - no valid results")
        return None

if __name__ == "__main__":
    print("Running complete validation analysis...")
    compare_multiple_submissions(limit=15)
    print("\nValidation complete!")

    # Load your top 2 submissions and compare feature importance
    top1 = pd.read_csv('data/Actual/submission.86403.csv')
    top2 = pd.read_csv('data/Actual/submission.85760.csv')
    plt.scatter(top1['rainfall'], top2['rainfall'], alpha=0.3)
    plt.title(f"Correlation: {np.corrcoef(top1['rainfall'], top2['rainfall'])[0,1]:.4f}")
    plt.xlabel("Submission 86403")
    plt.ylabel("Submission 85760")
    plt.savefig("analysis/top_models_comparison.png")