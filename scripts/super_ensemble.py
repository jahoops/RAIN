import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from typing import List, Dict, Tuple
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def create_optimized_blend(use_bootstrap_data: bool = True) -> pd.DataFrame:
    """
    Create an optimized blend using learned weights from synthetic data
    and incorporating bootstrap data if available.
    
    Args:
        use_bootstrap_data: Whether to use synthetic bootstrap data for calibration
        
    Returns:
        DataFrame with optimized blend submission
    """
    print("\n===== CREATING SUPER ENSEMBLE BLEND =====")
    
    # Step 1: Load available submissions from keep directory
    keep_dir = os.path.join('data', 'keep')
    if not os.path.exists(keep_dir):
        print("Keep directory not found! Creating it...")
        os.makedirs(keep_dir)
        print("Please place trusted submissions in the keep directory first.")
        return None
    
    # Find all CSV files in keep directory
    submission_files = [f for f in os.listdir(keep_dir) 
                        if f.endswith('.csv') and f.startswith('submission_')]
    
    if len(submission_files) < 2:
        print(f"Not enough submissions found in {keep_dir}. Need at least 2 files.")
        return None
    
    print(f"Found {len(submission_files)} trusted submissions to blend:")
    for file in submission_files:
        print(f"- {file}")
    
    # Step 2: Load the submissions
    submissions = []
    for file in submission_files:
        try:
            path = os.path.join(keep_dir, file)
            df = pd.read_csv(path)
            if 'id' in df.columns and 'rainfall' in df.columns:
                submissions.append({
                    'name': file,
                    'df': df,
                    'path': path
                })
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
    
    if len(submissions) < 2:
        print("Not enough valid submissions loaded. Need at least 2.")
        return None
    
    # Step 3: Find common IDs across all submissions
    common_ids = set(submissions[0]['df']['id'])
    for sub in submissions[1:]:
        common_ids = common_ids.intersection(set(sub['df']['id']))
    
    print(f"Found {len(common_ids)} common IDs across all submissions")
    
    # Step 4: Create a dataframe for blending
    blend_df = pd.DataFrame({'id': list(common_ids)})
    blend_df = blend_df.sort_values('id').reset_index(drop=True)
    
    # Add each submission's predictions
    for sub in submissions:
        sub_df = sub['df']
        common_sub_df = sub_df[sub_df['id'].isin(common_ids)].copy()
        common_sub_df = common_sub_df.sort_values('id').reset_index(drop=True)
        blend_df[sub['name']] = common_sub_df['rainfall'].values
    
    # Step 5: Load bootstrap data if available and requested
    bootstrap_features = None
    if use_bootstrap_data:
        bootstrap_path = os.path.join('data', 'synthetic_bootstrap.csv')
        if os.path.exists(bootstrap_path):
            try:
                bootstrap_df = pd.read_csv(bootstrap_path)
                if 'id' in bootstrap_df.columns:
                    print(f"Loaded bootstrap data with {len(bootstrap_df)} rows and {bootstrap_df.shape[1]} columns")
                    
                    # Find common IDs with our blend dataframe
                    common_bootstrap_ids = set(blend_df['id']).intersection(set(bootstrap_df['id']))
                    print(f"Found {len(common_bootstrap_ids)} common IDs with bootstrap data")
                    
                    if len(common_bootstrap_ids) > 0:
                        # Extract features from bootstrap data
                        bootstrap_subset = bootstrap_df[bootstrap_df['id'].isin(common_bootstrap_ids)]
                        feature_cols = [col for col in bootstrap_df.columns 
                                       if col != 'id' and col != 'rainfall']
                        
                        if len(feature_cols) > 0:
                            # Merge with blend dataframe
                            bootstrap_features = bootstrap_subset[['id'] + feature_cols]
                            blend_df = pd.merge(blend_df, bootstrap_features, on='id', how='left')
                            print(f"Added {len(feature_cols)} bootstrap features to blend")
            except Exception as e:
                print(f"Error loading bootstrap data: {str(e)}")
                bootstrap_features = None
    
    # Step 6: Determine optimal weights using available ground truth
    # If we have synthetic data with known values, we can learn optimal weights
    have_synthetic_labels = False
    synthetic_path = os.path.join('data', 'improved_synthetic_test.csv')
    
    if os.path.exists(synthetic_path):
        try:
            synthetic_df = pd.read_csv(synthetic_path)
            if 'id' in synthetic_df.columns and 'rainfall' in synthetic_df.columns:
                # Find common IDs with our blend dataframe
                common_synthetic_ids = set(blend_df['id']).intersection(set(synthetic_df['id']))
                print(f"Found {len(common_synthetic_ids)} common IDs with synthetic data")
                
                if len(common_synthetic_ids) >= 50:  # Enough data to learn weights
                    have_synthetic_labels = True
                    
                    # Create training data for weight optimization
                    train_df = blend_df[blend_df['id'].isin(common_synthetic_ids)].copy()
                    train_df = train_df.sort_values('id').reset_index(drop=True)
                    
                    # Get labels
                    synthetic_subset = synthetic_df[synthetic_df['id'].isin(common_synthetic_ids)]
                    synthetic_subset = synthetic_subset.sort_values('id').reset_index(drop=True)
                    train_df['true_rainfall'] = synthetic_subset['rainfall'].values
                    
                    # Train a model to learn optimal weights
                    print("\nLearning optimal weights from synthetic data:")
                    optimal_weights = learn_optimal_weights(train_df)
                    
                    # Print weights
                    print("\nOptimal submission weights:")
                    pred_columns = [col for col in train_df.columns 
                                   if col.startswith('submission_') and col.endswith('.csv')]
                    
                    for col, weight in zip(pred_columns, optimal_weights):
                        print(f"{col}: {weight:.6f}")
        except Exception as e:
            print(f"Error using synthetic data for weight optimization: {str(e)}")
            have_synthetic_labels = False
    
    # Step 7: Create blend variants
    
    # 7.1 Simple average blend (fallback)
    pred_columns = [col for col in blend_df.columns 
                   if col.startswith('submission_') and col.endswith('.csv')]
    
    blend_df['average_blend'] = blend_df[pred_columns].mean(axis=1)
    
    # 7.2 Weighted blend if we learned weights
    if have_synthetic_labels:
        blend_df['weighted_blend'] = create_weighted_prediction(blend_df, optimal_weights)
        preferred_blend = 'weighted_blend'
    else:
        # 7.3 If no synthetic labels, use corrcoef-weighted approach
        blend_df['corr_weighted_blend'] = create_correlation_weighted_blend(blend_df)
        preferred_blend = 'corr_weighted_blend'
    
    # Use the preferred blend
    best_blend_column = preferred_blend if preferred_blend in blend_df.columns else 'average_blend'
    print(f"\nUsing '{best_blend_column}' as the final blend")
    
    # Step 8: Create final submission
    submission_df = pd.DataFrame({
        'id': blend_df['id'],
        'rainfall': blend_df[best_blend_column]
    })
    
    # Step 9: Save submission file
    output_path = 'data/submission_super_ensemble.csv'
    submission_df.to_csv(output_path, index=False)
    print(f"\nSaved super ensemble submission to {output_path}")
    
    # Copy to keep directory for future use
    keep_path = os.path.join(keep_dir, 'submission_super_ensemble.csv')
    submission_df.to_csv(keep_path, index=False)
    print(f"Saved copy to keep directory: {keep_path}")
    
    # Step 10: Validate if possible
    try:
        from validate_submission import validate_submission
        print("\nValidating super ensemble submission:")
        result = validate_submission(output_path, quiet=False)
        
        if result and 'estimated_score' in result:
            print(f"\nEstimated contest score: {result['estimated_score']:.6f}")
            if float(result['estimated_score']) > 0.864:
                print("NEW HIGH SCORE LIKELY - This should beat the previous record of 0.86403!")
    except Exception as e:
        print(f"\nError validating submission: {str(e)}")
    
    return submission_df

def learn_optimal_weights(train_df: pd.DataFrame) -> np.ndarray:
    """Learn optimal weights for submissions using Ridge regression"""
    # Extract feature columns (submission predictions)
    pred_columns = [col for col in train_df.columns 
                   if col.startswith('submission_') and col.endswith('.csv')]
    
    # Extract target
    X = train_df[pred_columns].values
    y = train_df['true_rainfall'].values
    
    # Add a small amount of regularization for stability
    model = Ridge(alpha=0.01, fit_intercept=False, positive=True)
    
    # Fit model
    model.fit(X, y)
    
    # Get weights
    weights = model.coef_
    
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)
    
    return weights

def create_weighted_prediction(df: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    """Create weighted prediction using learned weights"""
    pred_columns = [col for col in df.columns 
                   if col.startswith('submission_') and col.endswith('.csv')]
    
    weighted_preds = np.zeros(len(df))
    
    for col, weight in zip(pred_columns, weights):
        weighted_preds += df[col].values * weight
    
    return weighted_preds

def create_correlation_weighted_blend(df: pd.DataFrame) -> np.ndarray:
    """Create a blend weighted by inter-submission correlations"""
    pred_columns = [col for col in df.columns 
                   if col.startswith('submission_') and col.endswith('.csv')]
    
    # Calculate correlation matrix between submissions
    corr_matrix = df[pred_columns].corr()
    
    # Average correlation of each submission with all others
    avg_corr = corr_matrix.mean()
    
    # Higher correlation = more similar to consensus = higher weight
    weights = avg_corr.values
    
    # Normalize to sum to 1
    weights = weights / np.sum(weights)
    
    print("\nCorrelation-based weights:")
    for col, weight in zip(pred_columns, weights):
        print(f"{col}: {weight:.6f}")
    
    # Create weighted prediction
    weighted_preds = np.zeros(len(df))
    for col, weight in zip(pred_columns, weights):
        weighted_preds += df[col].values * weight
    
    return weighted_preds

def explore_submissions(blend_report=True):
    """Explore and analyze all available submissions"""
    print("\n===== SUBMISSION ANALYSIS =====")
    
    # Create directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Look for submissions in data directory and keep directory
    all_submissions = []
    
    # Load from keep directory first (trusted submissions)
    keep_dir = os.path.join('data', 'keep')
    if os.path.exists(keep_dir):
        for file in os.listdir(keep_dir):
            if file.endswith('.csv') and file.startswith('submission_'):
                path = os.path.join(keep_dir, file)
                try:
                    df = pd.read_csv(path)
                    if 'id' in df.columns and 'rainfall' in df.columns:
                        all_submissions.append({
                            'name': file,
                            'df': df,
                            'path': path,
                            'trusted': True
                        })
                        print(f"Loaded trusted submission: {file}")
                except Exception as e:
                    print(f"Error loading {file}: {str(e)}")
    
    # Load from data directory
    for file in os.listdir('data'):
        if file.endswith('.csv') and file.startswith('submission_'):
            path = os.path.join('data', file)
            # Skip if already loaded from keep directory
            if any(sub['path'] == path for sub in all_submissions):
                continue
                
            try:
                df = pd.read_csv(path)
                if 'id' in df.columns and 'rainfall' in df.columns:
                    all_submissions.append({
                        'name': file,
                        'df': df,
                        'path': path,
                        'trusted': False
                    })
                    print(f"Loaded submission: {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    
    print(f"\nLoaded {len(all_submissions)} submission files")
    
    if not all_submissions:
        print("No submissions found to analyze.")
        return
    
    # Find common IDs across all submissions
    common_ids = set(all_submissions[0]['df']['id'])
    for sub in all_submissions[1:]:
        common_ids = common_ids.intersection(set(sub['df']['id']))
    
    print(f"Found {len(common_ids)} common IDs across all submissions")
    
    if len(common_ids) < 10:
        print("Too few common IDs for meaningful analysis.")
        return
    
    # Create analysis dataframe
    analysis_df = pd.DataFrame({'id': list(common_ids)})
    analysis_df = analysis_df.sort_values('id').reset_index(drop=True)
    
    # Add each submission's predictions
    for sub in all_submissions:
        sub_df = sub['df']
        common_sub_df = sub_df[sub_df['id'].isin(common_ids)].copy()
        common_sub_df = common_sub_df.sort_values('id').reset_index(drop=True)
        analysis_df[sub['name']] = common_sub_df['rainfall'].values
    
    # Calculate correlation matrix
    submission_cols = [sub['name'] for sub in all_submissions]
    corr_matrix = analysis_df[submission_cols].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='viridis', vmin=0.9, vmax=1.0,
                square=True, linewidths=.5, annot=True, fmt='.3f')
    plt.title('Submission Correlation Matrix')
    plt.tight_layout()
    plt.savefig('plots/submission_correlations.png')
    print("Saved correlation heatmap to plots/submission_correlations.png")
    
    # Generate blend report if requested
    if blend_report:
        high_corr_submissions = []
        low_corr_submissions = []
        trusted_submissions = []
        
        # Group submissions
        for i, sub in enumerate(all_submissions):
            # Calculate average correlation with other submissions
            sub_corr = corr_matrix.iloc[i].mean()
            
            if sub['trusted']:
                trusted_submissions.append(sub['name'])
            
            if sub_corr > 0.99:  # High correlation threshold
                high_corr_submissions.append(sub['name'])
            elif sub_corr < 0.95:  # Low correlation threshold
                low_corr_submissions.append(sub['name'])
        
        print("\n===== BLEND RECOMMENDATIONS =====")
        print(f"Trusted submissions: {len(trusted_submissions)}")
        
        # If we have at least one trusted submission with lower correlation
        if trusted_submissions and len(trusted_submissions) >= 2:
            print(f"Recommendation: Blend the {len(trusted_submissions)} trusted submissions")
            print("The trusted submissions provide enough diversity for a good blend.")
        elif high_corr_submissions and low_corr_submissions and trusted_submissions:
            print("Recommendation: Blend one submission from high-correlation group and one from low-correlation group")
            print("Plus all trusted submissions.")
            print("This maximizes diversity in the blend.")
        else:
            print("Recommendation: Use a weighted blend of all available submissions.")
    
    return analysis_df

if __name__ == "__main__":
    # First analyze existing submissions
    analysis_df = explore_submissions(blend_report=True)
    
    # Then create super ensemble
    submission = create_optimized_blend(use_bootstrap_data=True)
    
    print("\nProcess complete! Your super ensemble submission is ready.")
    print("For best results, validate against synthetic data and submit:")
    print("data/submission_super_ensemble.csv")
