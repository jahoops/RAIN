import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def create_recommended_ensemble():
    """Create an ensemble based on the filter_submissions.py recommendations"""
    print("=== BUILDING RECOMMENDED ENSEMBLE ===")
    
    # Specified models from filtered submissions recommendations
    recommended_models = {
        'submission_s_e_fixed.csv': 0.40,  # Best model with higher weight
        'submission_all_catboost_score_0.880755_fixed_fixed3.csv': 0.20,
        'submission_all_xgb_high_reg_score_0.878679_fixed3.csv': 0.20,
        'submission_ngboost_fixed.csv': 0.20
    }
    
    # Verify weights sum to 1
    total_weight = sum(recommended_models.values())
    if abs(total_weight - 1.0) > 0.001:
        print(f"WARNING: Weights don't sum to 1.0 (sum: {total_weight}). Normalizing...")
        for model in recommended_models:
            recommended_models[model] /= total_weight
    
    # Create directory for analysis
    os.makedirs('analysis', exist_ok=True)
    
    # Load each model submission
    keep_dir = 'data/keep'
    model_dfs = {}
    all_ids = set()
    
    for model_file, weight in recommended_models.items():
        model_path = os.path.join(keep_dir, model_file)
        if not os.path.exists(model_path):
            print(f"WARNING: {model_file} not found in keep directory")
            # Try to find in data directory
            model_path = os.path.join('data', model_file)
            if not os.path.exists(model_path):
                print(f"ERROR: {model_file} not found. Skipping...")
                continue
        
        try:
            df = pd.read_csv(model_path)
            print(f"Loaded {model_file} with {len(df)} rows")
            model_dfs[model_file] = df
            all_ids.update(set(df['id']))
        except Exception as e:
            print(f"Error loading {model_file}: {str(e)}")
    
    if not model_dfs:
        print("No models could be loaded. Cannot create ensemble.")
        return None
    
    # Find common IDs across all models
    common_ids = set()
    for i, (model, df) in enumerate(model_dfs.items()):
        if i == 0:
            common_ids = set(df['id'])
        else:
            common_ids.intersection_update(set(df['id']))
    
    print(f"Found {len(common_ids)} IDs common to all models")
    print(f"Total unique IDs across all models: {len(all_ids)}")
    
    if not common_ids:
        print("No common IDs found across all models. Using union of all IDs instead.")
        # Create a dataframe with all IDs
        ensemble_df = pd.DataFrame({'id': list(all_ids)})
        
        # Merge each model's predictions
        for model, df in model_dfs.items():
            ensemble_df = ensemble_df.merge(df, on='id', how='left', suffixes=('', f'_{model}'))
        
        # The first merge won't have a suffix, but subsequent ones will
        if 'rainfall' in ensemble_df.columns:
            ensemble_df.rename(columns={'rainfall': 'rainfall_first'}, inplace=True)
        
        # Fill missing values with median of existing predictions
        rainfall_cols = [col for col in ensemble_df.columns if col.startswith('rainfall_')]
        for col in rainfall_cols:
            median_val = ensemble_df[col].median()
            ensemble_df[col] = ensemble_df[col].fillna(median_val)
        
        # Create weighted ensemble
        ensemble_df['rainfall'] = 0
        for i, (model, weight) in enumerate(recommended_models.items()):
            col_name = 'rainfall_first' if i == 0 else f'rainfall_{model}'
            if col_name in ensemble_df.columns:
                ensemble_df['rainfall'] += weight * ensemble_df[col_name]
        
    else:
        # Filter each model to common IDs
        filtered_dfs = {}
        for model, df in model_dfs.items():
            filtered_dfs[model] = df[df['id'].isin(common_ids)].sort_values('id')
        
        # Create ensemble dataframe
        ensemble_df = pd.DataFrame({'id': sorted(list(common_ids))})
        
        # Add each model's predictions
        for model, df in filtered_dfs.items():
            ensemble_df = ensemble_df.merge(df[['id', 'rainfall']], on='id', how='left', suffixes=('', f'_{model}'))
        
        # The first merge won't have a suffix
        if len(filtered_dfs) > 1:
            first_model = next(iter(filtered_dfs.keys()))
            ensemble_df.rename(columns={'rainfall': f'rainfall_{first_model}'}, inplace=True)
        
        # Create weighted ensemble
        ensemble_df['rainfall'] = 0
        for model, weight in recommended_models.items():
            if f'rainfall_{model}' in ensemble_df.columns:
                ensemble_df['rainfall'] += weight * ensemble_df[f'rainfall_{model}']
        
    # Check for any NaN values
    if ensemble_df['rainfall'].isna().any():
        print("WARNING: Found NaN values in ensemble. Filling with 0.5...")
        ensemble_df['rainfall'] = ensemble_df['rainfall'].fillna(0.5)
    
    # Ensure values are in valid range
    ensemble_df['rainfall'] = np.clip(ensemble_df['rainfall'], 0, 1)
    
    # Create final submission
    final_submission = ensemble_df[['id', 'rainfall']]
    output_path = 'data/submission_recommended_ensemble.csv'
    final_submission.to_csv(output_path, index=False)
    print(f"Saved recommended ensemble to {output_path}")
    
    # Also save to keep directory
    keep_path = os.path.join(keep_dir, 'submission_recommended_ensemble.csv')
    final_submission.to_csv(keep_path, index=False)
    print(f"Also saved to {keep_path}")
    
    # Save to final directory for submission
    final_dir = 'data/final'
    os.makedirs(final_dir, exist_ok=True)
    final_path = os.path.join(final_dir, 'submission_final.csv')
    final_submission.to_csv(final_path, index=False)
    print(f"Saved as final submission to {final_path}")
    
    # Create distribution plot
    plt.figure(figsize=(12, 8))
    
    # Plot distributions of individual models and ensemble
    for model, df in model_dfs.items():
        model_name = model.replace('submission_', '').replace('.csv', '')
        sns.kdeplot(df['rainfall'], label=model_name, alpha=0.5)
    
    sns.kdeplot(final_submission['rainfall'], label='Final Ensemble', color='black', linewidth=2)
    
    plt.title('Prediction Distributions')
    plt.xlabel('Rainfall Prediction')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('analysis/ensemble_distributions.png')
    print("Saved distribution plot to analysis/ensemble_distributions.png")
    
    # Validate the submission if possible
    try:
        from validation_fixed import validate_submission
        print("\nValidating ensemble against synthetic data...")
        result = validate_submission(final_path)
        if result and 'estimated_score' in result:
            print(f"Estimated score: {result['estimated_score']:.6f}")
    except Exception as e:
        print(f"Could not validate submission: {e}")
    
    return final_submission

if __name__ == "__main__":
    print("Creating recommended ensemble based on filtering results...")
    create_recommended_ensemble()
    print("\nEnsemble creation complete!")
