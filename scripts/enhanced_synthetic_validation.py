import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

def generate_realistic_synthetic_data(submission_df, X_test_features):
    """Generate more realistic synthetic data based on feature relationships"""
    print("\n=== GENERATING ENHANCED SYNTHETIC DATA ===")
    
    # Training data to learn feature relationships
    from data_pipeline import create_complete_data_pipeline
    X_train_temporal, _ = create_complete_data_pipeline()
    
    y_train = X_train_temporal['rainfall']
    X_train = X_train_temporal.drop(['id', 'rainfall'], axis=1)
    
    # Preserve common columns only
    common_columns = list(set(X_train.columns).intersection(set(X_test_features.columns)))
    print(f"Using {len(common_columns)} common features to generate synthetic data")
    
    # Train model to generate synthetic values
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train[common_columns], y_train)
    
    # Generate predictions on test features as synthetic "ground truth"
    synthetic_rainfall = model.predict(X_test_features[common_columns])
    
    # Add some noise to avoid perfect correlation with our own predictions
    np.random.seed(42) 
    noise = np.random.normal(0, 0.05, len(synthetic_rainfall))
    synthetic_rainfall = np.clip(synthetic_rainfall + noise, 0, 1)
    
    # Create dataframe with submission IDs
    synthetic_df = pd.DataFrame({
        'id': submission_df['id'],
        'rainfall': synthetic_rainfall
    })
    
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
    
    # Load test features
    test_features_path = 'data/test_features.csv'
    if not os.path.exists(test_features_path):
        print(f"ERROR: Could not find test features at {test_features_path}")
        return None
        
    X_test_features = pd.read_csv(test_features_path)
    
    # Generate enhanced synthetic data
    synthetic_df = generate_realistic_synthetic_data(submission_df, X_test_features)
    
    # Calculate metrics
    predictions = submission_df['rainfall'].values
    ground_truth = synthetic_df['rainfall'].values
    
    # Standard metrics
    mse = mean_squared_error(ground_truth, predictions)
    correlation = np.corrcoef(ground_truth, predictions)[0, 1]
    
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
    
    # Estimated leaderboard score (based on your correlation with real scores)
    estimated_lb = (1 - mse) * 0.95  # Adjust this coefficient based on your observations
    print(f"Estimated leaderboard score: {estimated_lb:.6f}")
    
    # Visualization for better understanding
    os.makedirs('analysis/validation', exist_ok=True)
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.scatter(predictions, ground_truth, alpha=0.3)
    plt.xlabel('Predictions')
    plt.ylabel('Synthetic Ground Truth')
    plt.title(f'Correlation: {correlation:.4f}')
    
    plt.subplot(2, 2, 2)
    sns.kdeplot(predictions, label='Predictions')
    sns.kdeplot(ground_truth, label='Synthetic Ground Truth')
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
    
    # QQ plot of residuals
    plt.subplot(2, 2, 4)
    sns.histplot(residuals, kde=True)
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

# Update the submission directory to match your actual candidate submissions path

def compare_multiple_submissions():
    """Compare multiple submissions to identify best model"""
    print("\n=== COMPARING MULTIPLE SUBMISSIONS ===")
    
    # Find all submission files - TRY BOTH PATHS
    submission_dirs = ['data/candidates', 'data/Actual']
    
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
    
    print(f"Found {len(submission_files)} submission files in {used_dir}")
    
    # Validate each submission
    results = []
    for file in submission_files:
        path = os.path.join(used_dir, file)
        print(f"\nValidating {file}...")
        metrics = validate_submission(path, 'both')
        if metrics:
            results.append({
                'file': file,
                'mse': metrics['mse'],
                'correlation': metrics['correlation'], 
                'auc': metrics['auc'],
                'estimated_lb': (1 - metrics['mse']) * 0.95
            })
    
    # Create a sorted dataframe of results
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('estimated_lb', ascending=False)
        
        print("\n=== FINAL COMPARISON ===")
        print(results_df)
        
        # Save results
        results_df.to_csv('analysis/validation/submission_comparison.csv', index=False)
        print("Saved comparison results to analysis/validation/submission_comparison.csv")
        
        # Plot comparative performance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='file', y='estimated_lb', data=results_df.head(10))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('analysis/validation/submission_comparison.png')
        
        return results_df
    else:
        print("Could not compare submissions - no valid results")
        return None

if __name__ == "__main__":
    print("Running enhanced synthetic validation...")
    compare_multiple_submissions()
    print("\nValidation complete!")