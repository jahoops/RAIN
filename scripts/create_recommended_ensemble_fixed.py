import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# Try to use the fixed pandas compatibility
try:
    from utils.pandas_compat_fixed import fix_pandas_compatibility, monkey_patch_seaborn
    restore_pandas = fix_pandas_compatibility()
    restore_seaborn = monkey_patch_seaborn()
    print("Applied pandas compatibility fixes")
    import seaborn as sns
except ImportError:
    print("Could not apply pandas compatibility fixes - using matplotlib only")
    # No seaborn import

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
    
    # Create empty ensemble dataframe with all unique IDs
    ensemble_df = pd.DataFrame({'id': sorted(list(all_ids))})
    
    # Add each model's predictions
    for i, (model_name, df) in enumerate(model_dfs.items()):
        col_name = f'rainfall_{i}'
        ensemble_df = ensemble_df.merge(
            df[['id', 'rainfall']].rename(columns={'rainfall': col_name}),
            on='id', 
            how='left'
        )
    
    # Identify which models contributed to each ID
    rainfall_cols = [col for col in ensemble_df.columns if col.startswith('rainfall_')]
    
    # Calculate weighted ensemble
    ensemble_df['rainfall'] = 0
    weights_used = {}
    
    # Apply each model with its weight
    for i, (model_name, weight) in enumerate(recommended_models.items()):
        col_name = f'rainfall_{i}'
        if col_name in ensemble_df.columns:
            # Apply weight only where the model has predictions
            valid_mask = ensemble_df[col_name].notna()
            ensemble_df.loc[valid_mask, 'rainfall'] += weight * ensemble_df.loc[valid_mask, col_name]
            
            # Track what weights were applied
            for idx in ensemble_df.index[valid_mask]:
                if idx not in weights_used:
                    weights_used[idx] = 0
                weights_used[idx] += weight
    
    # Normalize the weights for each row
    for idx, total_weight in weights_used.items():
        if total_weight > 0 and total_weight != 1.0:
            ensemble_df.loc[idx, 'rainfall'] /= total_weight
    
    # Fill any rows that got no weights with median value
    no_weights = set(ensemble_df.index) - set(weights_used.keys())
    if no_weights:
        print(f"WARNING: {len(no_weights)} rows had no model predictions. Filling with median.")
        median_val = ensemble_df['rainfall'].median()
        ensemble_df.loc[list(no_weights), 'rainfall'] = median_val
    
    # Check for any NaN values
    if ensemble_df['rainfall'].isna().any():
        print(f"WARNING: Found {ensemble_df['rainfall'].isna().sum()} NaN values in ensemble. Filling with 0.5...")
        ensemble_df['rainfall'] = ensemble_df['rainfall'].fillna(0.5)
    
    # Ensure values are in valid range
    ensemble_df['rainfall'] = np.clip(ensemble_df['rainfall'], 0, 1)
    
    # Create final submission
    final_submission = ensemble_df[['id', 'rainfall']]
    output_path = 'data/submission_recommended_ensemble_fixed.csv'
    final_submission.to_csv(output_path, index=False)
    print(f"Saved recommended ensemble to {output_path}")
    
    # Also save to keep directory
    keep_path = os.path.join(keep_dir, 'submission_recommended_ensemble_fixed.csv')
    final_submission.to_csv(keep_path, index=False)
    print(f"Also saved to {keep_path}")
    
    # Save to final directory for submission
    final_dir = 'data/final'
    os.makedirs(final_dir, exist_ok=True)
    final_path = os.path.join(final_dir, 'submission_final.csv')
    final_submission.to_csv(final_path, index=False)
    print(f"Saved as final submission to {final_path}")
    
    # Create distribution plot using pure matplotlib (avoid seaborn KDE issues)
    plt.figure(figsize=(12, 8))
    
    # Plot histograms instead of KDE plots
    for i, (model, df) in enumerate(model_dfs.items()):
        model_name = model.replace('submission_', '').replace('.csv', '')
        plt.hist(df['rainfall'], bins=30, alpha=0.3, label=model_name, density=True)
    
    # Add the ensemble
    plt.hist(final_submission['rainfall'], bins=30, alpha=0.5, 
             label='Final Ensemble', color='black', density=True)
    
    plt.title('Prediction Distributions')
    plt.xlabel('Rainfall Prediction')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('analysis/ensemble_distributions_fixed.png')
    print("Saved distribution plot to analysis/ensemble_distributions_fixed.png")
    
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
