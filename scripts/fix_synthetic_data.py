import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def diagnose_synthetic_data():
    """Diagnose issues with synthetic data"""
    print("=== DIAGNOSING SYNTHETIC DATA ISSUES ===")
    
    # Load synthetic data
    synthetic_path = 'data/synthetic_aligned.csv'
    if not os.path.exists(synthetic_path):
        print(f"Synthetic data not found at {synthetic_path}")
        return None
        
    synthetic_df = pd.read_csv(synthetic_path)
    print(f"Loaded synthetic data with {len(synthetic_df)} rows")
    
    # Check basic statistics
    print("\nSynthetic data statistics:")
    print(f"Mean: {synthetic_df['rainfall'].mean():.4f}")
    print(f"Median: {synthetic_df['rainfall'].median():.4f}")
    print(f"Std: {synthetic_df['rainfall'].std():.4f}")
    print(f"Min: {synthetic_df['rainfall'].min():.4f}")
    print(f"Max: {synthetic_df['rainfall'].max():.4f}")
    
    # Check distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(synthetic_df['rainfall'], kde=True)
    plt.title('Synthetic Data Distribution')
    plt.xlabel('Rainfall')
    
    # Check ID continuity
    plt.subplot(1, 2, 2)
    plt.scatter(synthetic_df['id'], synthetic_df['rainfall'], alpha=0.5, s=10)
    plt.title('Rainfall vs ID')
    plt.xlabel('ID')
    plt.ylabel('Rainfall')
    
    plt.tight_layout()
    plt.savefig('analysis/synthetic_data_analysis.png')
    
    # Check if synthetic data has negative values or values > 1
    abnormal = ((synthetic_df['rainfall'] < 0) | (synthetic_df['rainfall'] > 1)).sum()
    if abnormal > 0:
        print(f"WARNING: Found {abnormal} values outside the 0-1 range!")
        print("This suggests the synthetic data is not properly normalized for classification.")
    
    # Create corrected synthetic data
    print("\n=== CREATING IMPROVED SYNTHETIC DATA ===")
    
    # Determine if data looks like classification (bimodal) or regression
    rainfall = synthetic_df['rainfall']
    is_bimodal = False
    
    # Simple bimodality check - look for concentration of values near 0 and 1
    counts_near_0 = ((rainfall >= -0.1) & (rainfall <= 0.1)).sum()
    counts_near_1 = ((rainfall >= 0.9) & (rainfall <= 1.1)).sum()
    
    if (counts_near_0 + counts_near_1) > len(rainfall) * 0.3:
        is_bimodal = True
        print("Data appears to have bimodal distribution consistent with classification task")
    else:
        print("Data appears to have continuous distribution consistent with regression task")
    
    # Create normalized version for classification
    fixed_synthetic = synthetic_df.copy()
    
    # Clip values to 0-1 range
    fixed_synthetic['rainfall'] = np.clip(fixed_synthetic['rainfall'], 0, 1)
    
    # If bimodal, binarize further
    if is_bimodal:
        # Convert to binary 0/1 based on threshold
        threshold = 0.5
        fixed_synthetic['rainfall_binary'] = (fixed_synthetic['rainfall'] > threshold).astype(float)
    
    # Save fixed synthetic data
    fixed_path = 'data/synthetic_fixed.csv'
    fixed_synthetic.to_csv(fixed_path, index=False)
    print(f"Saved normalized synthetic data to {fixed_path}")
    
    return fixed_synthetic

def select_best_model():
    """Select the best model based on intrinsic properties"""
    print("\n=== SELECTING BEST MODEL ===")
    
    # Load all submissions from keep directory
    keep_dir = 'data/keep'
    submission_files = glob.glob(os.path.join(keep_dir, 'submission_*.csv'))
    
    if not submission_files:
        print("No submission files found in keep directory")
        return None
    
    # Load fixed synthetic data
    try:
        fixed_synthetic = pd.read_csv('data/synthetic_fixed.csv')
        print(f"Loaded fixed synthetic data with {len(fixed_synthetic)} rows")
        has_synthetic = True
    except:
        print("Could not load fixed synthetic data. Proceeding without it.")
        has_synthetic = False
    
    # Evaluate model properties
    model_stats = []
    for file_path in submission_files:
        model_name = os.path.basename(file_path).replace('submission_', '').replace('.csv', '')
        try:
            df = pd.read_csv(file_path)
            
            # Basic statistics
            mean = df['rainfall'].mean()
            median = df['rainfall'].median()
            std = df['rainfall'].std()
            min_val = df['rainfall'].min()
            max_val = df['rainfall'].max()
            
            # Check if predictions are all unique (important for ranking)
            unique_ratio = len(df['rainfall'].unique()) / len(df['rainfall'])
            
            # Check if values are properly in 0-1 range
            normalized = ((df['rainfall'] >= 0) & (df['rainfall'] <= 1)).all()
            
            # Get correlation with fixed synthetic data if available
            correlation = np.nan
            if has_synthetic:
                common_ids = set(df['id']).intersection(set(fixed_synthetic['id']))
                if common_ids:
                    df_common = df[df['id'].isin(common_ids)].sort_values('id')
                    syn_common = fixed_synthetic[fixed_synthetic['id'].isin(common_ids)].sort_values('id')
                    if 'rainfall_binary' in fixed_synthetic.columns:
                        # Try binary evaluation if available
                        binary_targets = syn_common['rainfall_binary'].values
                        try:
                            auc = roc_auc_score(binary_targets, df_common['rainfall'])
                            correlation = 2 * (auc - 0.5)  # Scale from -1 to 1
                        except:
                            pass
                    
                    # Regular correlation
                    if np.isnan(correlation):
                        correlation = np.corrcoef(df_common['rainfall'], syn_common['rainfall'])[0, 1]
            
            # Calculate bimodality coefficient - higher means more bimodal
            skew = df['rainfall'].skew()
            kurt = df['rainfall'].kurtosis()
            bimodality = (skew**2 + 1) / kurt if kurt > 0 else 0
            
            # Calculate binary entropy - higher means more balanced binary predictions
            p = df['rainfall'].mean()
            binary_entropy = -p * np.log2(p+1e-10) - (1-p) * np.log2(1-p+1e-10)
            
            model_stats.append({
                'model': model_name,
                'file_path': file_path,
                'mean': mean,
                'median': median,
                'std': std,
                'min': min_val,
                'max': max_val,
                'unique_ratio': unique_ratio,
                'normalized': normalized,
                'correlation': correlation,
                'bimodality': bimodality,
                'binary_entropy': binary_entropy
            })
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    
    # Create a score for each model based on intrinsic properties
    for model in model_stats:
        quality_score = 0
        
        # Prefer models with predictions properly in 0-1 range
        quality_score += 10 if model['normalized'] else 0
        
        # Prefer models with higher variability in predictions
        quality_score += 5 * (model['std'] / 0.5) if model['std'] < 0.5 else 5
        
        # Prefer models with good spread between min and max
        range_score = min(1, (model['max'] - model['min'])) * 5
        quality_score += range_score
        
        # Prefer models with higher unique ratios (more granular predictions)
        quality_score += model['unique_ratio'] * 5
        
        # If we have correlation data, use it (but less weight due to synthetic issues)
        if not np.isnan(model['correlation']):
            corr_value = model['correlation'] 
            quality_score += max(0, corr_value) * 2
        
        model['quality_score'] = quality_score
    
    # Sort models by quality score
    model_stats.sort(key=lambda x: x['quality_score'], reverse=True)
    
    # Print top models
    print("\nTop models by quality score:")
    for i, model in enumerate(model_stats[:5]):
        print(f"{i+1}. {model['model']}: Score={model['quality_score']:.2f}, Mean={model['mean']:.4f}, Std={model['std']:.4f}")
    
    # Select best model
    best_model = model_stats[0]
    print(f"\nSelected best model: {best_model['model']} (score: {best_model['quality_score']:.2f})")
    
    # Load best model data
    best_df = pd.read_csv(best_model['file_path'])
    
    # Also create a simple average ensemble of top 3 models
    print("\nCreating ensemble of top 3 models...")
    ensemble_dfs = []
    
    for i in range(min(3, len(model_stats))):
        model = model_stats[i]
        df = pd.read_csv(model['file_path'])
        ensemble_dfs.append(df)
        print(f"Added {model['model']} to ensemble (weight: {1/3:.2f})")
    
    # Find common IDs across all models
    common_ids = set(ensemble_dfs[0]['id'])
    for df in ensemble_dfs[1:]:
        common_ids = common_ids.intersection(set(df['id']))
    
    # Create ensemble dataframe
    ensemble_df = pd.DataFrame({'id': sorted(list(common_ids))})
    
    # Add each model's predictions
    for i, df in enumerate(ensemble_dfs):
        df_filtered = df[df['id'].isin(common_ids)].sort_values('id')
        ensemble_df[f'model_{i}'] = df_filtered['rainfall'].values
    
    # Create simple average ensemble
    ensemble_df['rainfall'] = ensemble_df[[f'model_{i}' for i in range(len(ensemble_dfs))]].mean(axis=1)
    
    # Save best model and ensemble
    os.makedirs('data/final', exist_ok=True)
    
    best_path = 'data/final/submission_best_model.csv'
    best_df.to_csv(best_path, index=False)
    print(f"Saved best model to {best_path}")
    
    ensemble_path = 'data/final/submission_final.csv'
    ensemble_final = ensemble_df[['id', 'rainfall']]
    ensemble_final.to_csv(ensemble_path, index=False)
    print(f"Saved ensemble to {ensemble_path}")
    
    # Create visualization of best model and ensemble
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(best_df['rainfall'], kde=True, color='blue')
    plt.title(f'Best Model: {best_model["model"]}')
    plt.xlabel('Rainfall')
    
    plt.subplot(1, 2, 2)
    sns.histplot(ensemble_final['rainfall'], kde=True, color='green')
    plt.title('Top 3 Models Ensemble')
    plt.xlabel('Rainfall')
    
    plt.tight_layout()
    plt.savefig('analysis/best_models.png')
    print("Saved visualization to analysis/best_models.png")
    
    return ensemble_final

if __name__ == "__main__":
    print("Starting synthetic data diagnosis and model selection...")
    
    # Create analysis directory
    os.makedirs('analysis', exist_ok=True)
    
    # Fix synthetic data
    fixed_synthetic = diagnose_synthetic_data()
    
    # Select best model
    final_submission = select_best_model()
    
    print("\nProcess complete! Check data/final/submission_final.csv for your new submission.")
