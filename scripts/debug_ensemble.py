import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def debug_poor_ensemble_performance():
    """Debug and fix the poor ensemble performance"""
    print("=== DEBUGGING ENSEMBLE PERFORMANCE ISSUES ===")
    
    # Load the synthetic data for validation
    synthetic_path = 'data/synthetic_aligned.csv'
    if not os.path.exists(synthetic_path):
        print(f"ERROR: Synthetic validation data not found at {synthetic_path}")
        return None
    
    synthetic_df = pd.read_csv(synthetic_path)
    print(f"Loaded synthetic data with {len(synthetic_df)} rows")
    
    # Find all submissions in keep directory
    keep_dir = 'data/keep'
    submission_files = []
    for f in os.listdir(keep_dir):
        if f.startswith('submission_') and f.endswith('.csv'):
            submission_files.append(os.path.join(keep_dir, f))
    
    # Create directory for analysis
    os.makedirs('analysis/debug', exist_ok=True)
    
    # Analyze each submission against synthetic data
    results = []
    model_dfs = {}
    
    for file_path in submission_files:
        model_name = os.path.basename(file_path).replace('submission_', '').replace('.csv', '')
        
        try:
            df = pd.read_csv(file_path)
            # Store for later use
            model_dfs[model_name] = df
            
            # Find common IDs with synthetic data
            common_ids = set(df['id']).intersection(set(synthetic_df['id']))
            
            if len(common_ids) == 0:
                print(f"WARNING: {model_name} has no common IDs with synthetic data")
                continue
                
            # Filter to common IDs and sort
            df_common = df[df['id'].isin(common_ids)].sort_values('id')
            syn_common = synthetic_df[synthetic_df['id'].isin(common_ids)].sort_values('id')
            
            # Calculate correlation
            correlation = np.corrcoef(df_common['rainfall'], syn_common['rainfall'])[0, 1]
            
            # Calculate some stats
            mean = df['rainfall'].mean()
            std = df['rainfall'].std()
            min_val = df['rainfall'].min()
            max_val = df['rainfall'].max()
            
            # Create visualization of this model vs synthetic
            plt.figure(figsize=(10, 8))
            plt.scatter(df_common['rainfall'], syn_common['rainfall'], alpha=0.3)
            plt.xlabel(f'{model_name} Predictions')
            plt.ylabel('Synthetic Ground Truth')
            plt.title(f'{model_name} vs Synthetic Data (Correlation: {correlation:.4f})')
            plt.grid(True)
            plt.savefig(f'analysis/debug/{model_name}_vs_synthetic.png')
            plt.close()
            
            # Store results
            results.append({
                'model': model_name,
                'correlation': correlation,
                'mean': mean,
                'std': std,
                'min': min_val,
                'max': max_val,
                'common_ids': len(common_ids)
            })
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {str(e)}")
    
    # Sort by correlation
    results = sorted(results, key=lambda x: x['correlation'], reverse=True)
    
    # Print summary
    print("\n=== MODEL PERFORMANCE SUMMARY ===")
    print(f"{'Model':<40} {'Correlation':>12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 90)
    
    for res in results:
        print(f"{res['model']:<40} {res['correlation']:>12.4f} {res['mean']:>8.4f} {res['std']:>8.4f} {res['min']:>8.4f} {res['max']:>8.4f}")
    
    # Check if we have models with positive correlation
    positive_corr_models = [r for r in results if r['correlation'] > 0.1]
    
    if not positive_corr_models:
        print("\nWARNING: No models with positive correlation above 0.1!")
        print("This suggests a fundamental issue with either the synthetic data or all models.")
        
        # Check if the synthetic data looks reasonable
        print("\nSynthetic data statistics:")
        print(f"Mean: {synthetic_df['rainfall'].mean():.4f}")
        print(f"Std: {synthetic_df['rainfall'].std():.4f}")
        print(f"Min: {synthetic_df['rainfall'].min():.4f}")
        print(f"Max: {synthetic_df['rainfall'].max():.4f}")
        
        # Create histogram of synthetic data
        plt.figure(figsize=(10, 6))
        plt.hist(synthetic_df['rainfall'], bins=30)
        plt.title('Synthetic Data Distribution')
        plt.xlabel('Rainfall')
        plt.ylabel('Count')
        plt.savefig('analysis/debug/synthetic_distribution.png')
        plt.close()
        
        print("\nAnalysis suggests the synthetic data may not be appropriate for validation.")
        print("Attempting to create a better ensemble based on model diversity...")
    else:
        print(f"\nFound {len(positive_corr_models)} models with positive correlation.")
        print("Best model: " + positive_corr_models[0]['model'])
    
    # Try to create a better ensemble based on model diversity
    print("\n=== CREATING DIVERSITY-BASED ENSEMBLE ===")
    
    # Select top 5 models with highest standard deviation (most informative)
    diverse_models = sorted(results, key=lambda x: x['std'], reverse=True)[:5]
    print("Selected diverse models:")
    for model in diverse_models:
        print(f"- {model['model']} (std: {model['std']:.4f})")
    
    # Create ensemble from diverse models
    ensemble_ids = set()
    for model in diverse_models:
        model_df = model_dfs[model['model']]
        ensemble_ids.update(set(model_df['id']))
    
    ensemble_ids = sorted(list(ensemble_ids))
    print(f"Ensemble will cover {len(ensemble_ids)} IDs")
    
    # Create ensemble dataframe
    ensemble_df = pd.DataFrame({'id': ensemble_ids})
    
    # Add each model's predictions
    for model in diverse_models:
        model_name = model['model']
        model_df = model_dfs[model_name]
        
        ensemble_df = ensemble_df.merge(
            model_df[['id', 'rainfall']].rename(columns={'rainfall': model_name}),
            on='id',
            how='left'
        )
    
    # Fill missing values with column medians
    for model in diverse_models:
        col = model['model']
        if ensemble_df[col].isna().any():
            median = ensemble_df[col].median()
            ensemble_df[col].fillna(median, inplace=True)
    
    # Create simple average ensemble
    ensemble_df['rainfall'] = ensemble_df[[model['model'] for model in diverse_models]].mean(axis=1)
    
    # Create final submission
    final_submission = ensemble_df[['id', 'rainfall']]
    
    output_path = 'data/submission_diverse_ensemble.csv'
    final_submission.to_csv(output_path, index=False)
    
    # Also copy to final directory
    final_dir = 'data/final'
    os.makedirs(final_dir, exist_ok=True)
    final_path = os.path.join(final_dir, 'submission_final.csv')
    final_submission.to_csv(final_path, index=False)
    
    print(f"\nCreated diversity-based ensemble at {output_path}")
    print(f"Also saved as final submission to {final_path}")
    
    # Also try to use the single best model if available
    if positive_corr_models:
        best_model_name = positive_corr_models[0]['model']
        best_model_df = model_dfs[best_model_name]
        
        best_path = 'data/submission_best_model.csv'
        best_model_df.to_csv(best_path, index=False)
        print(f"\nSaved best model ({best_model_name}) to {best_path}")
        print("Consider using this as your final submission if the ensemble doesn't perform well.")
    
    return final_submission

if __name__ == "__main__":
    print("Debugging ensemble performance issues...")
    debug_poor_ensemble_performance()
    print("\nDebug analysis complete!")
