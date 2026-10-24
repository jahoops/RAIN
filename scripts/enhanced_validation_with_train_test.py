import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns

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
        print(f"WARNING: Only {len(matching_indices)} matching IDs found. Training validation may not be reliable.")
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

def compare_multiple_submissions(limit=None):
    """Compare multiple submissions with both synthetic and training validation"""
    print("\n=== COMPARING MULTIPLE SUBMISSIONS ===")
    
    # Find all submission files
    submission_dirs = ['data/Actual', 'data/candidates']
    
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
        
        if metrics:
            result = {
                'file': file,
                'mse': metrics['mse'],
                'correlation': metrics['correlation'], 
                'auc': metrics['auc'] if metrics['auc'] else 0.0,
                'estimated_lb': (1 - metrics['mse']) * 0.95
            }
            
            # Add training metrics if available
            if train_metrics:
                result.update({
                    'train_mse': train_metrics['train_mse'],
                    'train_correlation': train_metrics['train_correlation'],
                    'train_score': train_metrics['train_score']
                })
                
                # Extract score from filename if possible (for leaderboard correlation)
                try:
                    score_part = file.split('.')[-2]
                    if score_part.isdigit() and len(score_part) == 5:
                        lb_score = float(score_part) / 100000.0
                        result['lb_score'] = lb_score
                except:
                    pass
                
            results.append(result)
    
    # Create a sorted dataframe of results
    if results:
        results_df = pd.DataFrame(results)
        
        # Calculate correlation between synthetic validation and training validation
        if 'train_score' in results_df.columns and 'estimated_lb' in results_df.columns:
            correlation = np.corrcoef(results_df['train_score'], results_df['estimated_lb'])[0, 1]
            print(f"\nCorrelation between synthetic and training validation: {correlation:.4f}")
            
        # Calculate correlation with leaderboard score if available
        if 'lb_score' in results_df.columns and 'estimated_lb' in results_df.columns:
            lb_corr = np.corrcoef(results_df['lb_score'], results_df['estimated_lb'])[0, 1]
            print(f"Correlation between synthetic validation and leaderboard: {lb_corr:.4f}")
            
            # If correlation is decent, update our estimation formula
            if abs(lb_corr) > 0.5:
                print("Using leaderboard correlation to improve estimation formula")
                
                # Linear regression to map synthetic scores to leaderboard scores
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression()
                reg.fit(results_df[['estimated_lb']], results_df['lb_score'])
                
                # Apply the correction
                results_df['calibrated_lb'] = reg.predict(results_df[['estimated_lb']])
                print(f"Calibration: LB ≈ {reg.coef_[0]:.4f} × Synthetic + {reg.intercept_:.4f}")
        
        # Sort by the best available metric
        if 'calibrated_lb' in results_df.columns:
            results_df = results_df.sort_values('calibrated_lb', ascending=False)
        elif 'lb_score' in results_df.columns:
            results_df = results_df.sort_values('lb_score', ascending=False)
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
            plt.scatter(results_df['estimated_lb'], results_df['train_score'])
            plt.xlabel('Synthetic Validation Score')
            plt.ylabel('Training Validation Score')
            plt.title(f'Correlation: {correlation:.4f}')
        
        # Plot correlation with leaderboard if available
        plt.subplot(2, 2, 2)
        if 'lb_score' in results_df.columns and 'estimated_lb' in results_df.columns:
            plt.scatter(results_df['estimated_lb'], results_df['lb_score'])
            plt.xlabel('Synthetic Validation Score')
            plt.ylabel('Leaderboard Score')
            plt.title(f'LB Correlation: {lb_corr:.4f}')
            
            # Add the regression line
            if 'calibrated_lb' in results_df.columns:
                x = np.linspace(
                    results_df['estimated_lb'].min(),
                    results_df['estimated_lb'].max(),
                    100
                )
                plt.plot(x, reg.predict(x.reshape(-1, 1)), 'r--')
        
        # Plot top model performances
        plt.subplot(2, 1, 2)
        top_models = results_df.head(10)
        
        if 'calibrated_lb' in results_df.columns:
            plt.bar(top_models['file'], top_models['calibrated_lb'])
            plt.ylabel('Calibrated LB Score')
        elif 'lb_score' in results_df.columns:
            plt.bar(top_models['file'], top_models['lb_score'])
            plt.ylabel('Leaderboard Score')
        else:
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

# Keep the existing validate_submission function and add this at the end:
if __name__ == "__main__":
    print("Running enhanced validation with training data comparison...")
    compare_multiple_submissions(limit=15)
    print("\nValidation complete!")
