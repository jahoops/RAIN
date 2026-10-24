import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.metrics import roc_auc_score
import glob

def create_winning_ensemble():
    """Create an optimized ensemble of your best-performing models"""
    print("Creating winning ensemble from top models...")
    
    # Models to include in the ensemble
    models = [
        'neural',              # Regular neural network
        'synthetic_neural',    # Synthetic data neural network 
        'uncertainty_model',   # Uncertainty-aware model
        'lgb_fixed',           # Fixed LightGBM model
        'xgb_fixed',           # Fixed XGBoost model
    ]
    
    # Try to load these specific submissions from keep directory
    keep_dir = 'data/keep'
    submissions = {}
    
    for model_name in models:
        file_name = f'submission_{model_name}.csv'
        file_path = os.path.join(keep_dir, file_name)
        
        if os.path.exists(file_path):
            try:
                submissions[model_name] = pd.read_csv(file_path)
                print(f"Loaded {model_name} submission: {len(submissions[model_name])} rows")
            except Exception as e:
                print(f"Error loading {model_name}: {str(e)}")
    
    # Load u_b submission specifically (it seems to be a good performer)
    u_b_path = os.path.join(keep_dir, 'submission_u_b.csv')
    if os.path.exists(u_b_path):
        try:
            submissions['u_b'] = pd.read_csv(u_b_path)
            print(f"Loaded u_b submission: {len(submissions['u_b'])} rows")
        except:
            print("Could not load u_b submission")
    
    # If not enough models loaded, try to find any submissions in the keep dir
    if len(submissions) < 3:
        print("Not enough specified models found. Looking for any available submissions...")
        submission_files = glob.glob(os.path.join(keep_dir, 'submission_*.csv'))
        
        for file_path in submission_files:
            model_name = os.path.basename(file_path).replace('submission_', '').replace('.csv', '')
            if model_name not in submissions:
                try:
                    submissions[model_name] = pd.read_csv(file_path)
                    print(f"Added additional model {model_name}: {len(submissions[model_name])} rows")
                    
                    # Stop after adding enough models
                    if len(submissions) >= 5:
                        break
                except:
                    pass
    
    if len(submissions) < 2:
        print("Not enough submissions found to create an ensemble")
        return None
    
    # Find common IDs across all submissions
    common_ids = set(submissions[list(submissions.keys())[0]]['id'])
    for name, df in submissions.items():
        common_ids = common_ids.intersection(set(df['id']))
    
    common_ids = sorted(list(common_ids))
    print(f"Found {len(common_ids)} common IDs across {len(submissions)} submissions")
    
    if len(common_ids) == 0:
        print("No common IDs found across submissions. Cannot create ensemble.")
        return None
    
    # Create ensemble dataframe
    ensemble_df = pd.DataFrame({'id': common_ids})
    
    # Add predictions from each model
    for name, df in submissions.items():
        df_filtered = df[df['id'].isin(common_ids)].sort_values('id')
        ensemble_df[name] = df_filtered['rainfall'].values
    
    # Check correlation between model predictions
    corr = ensemble_df.drop('id', axis=1).corr()
    print("\nModel correlation matrix:")
    print(corr)
    
    # Define custom weights based on analysis:
    # - Give more weight to synthetic neural model (highest CV)
    # - Give more weight to u_b if available (appears to be good)
    weights = {}
    
    if 'synthetic_neural' in ensemble_df.columns:
        weights['synthetic_neural'] = 0.35  # Higher weight based on CV performance
    
    if 'neural' in ensemble_df.columns:
        weights['neural'] = 0.25  # Good CV performance
        
    if 'u_b' in ensemble_df.columns:
        weights['u_b'] = 0.20  # Suggested to be a good performer
        
    # Distribute remaining weight evenly among other models
    remaining_models = [model for model in ensemble_df.columns 
                       if model != 'id' and model not in weights]
    
    remaining_weight = 1.0 - sum(weights.values())
    for model in remaining_models:
        weights[model] = remaining_weight / len(remaining_models)
    
    # Print chosen weights
    print("\nEnsemble weights:")
    for model, weight in weights.items():
        print(f"{model}: {weight:.4f}")
    
    # Create weighted ensemble predictions
    ensemble_df['winning_blend'] = 0
    for model, weight in weights.items():
        if model in ensemble_df.columns:
            ensemble_df['winning_blend'] += ensemble_df[model] * weight
    
    # Create winning submission
    final_submission = pd.DataFrame({
        'id': ensemble_df['id'],
        'rainfall': ensemble_df['winning_blend']
    })
    
    # Save predictions
    output_path = 'data/submission_winning_blend.csv'
    final_submission.to_csv(output_path, index=False)
    print(f"Winning ensemble saved to {output_path}")
    
    # Copy to keep directory
    keep_path = os.path.join(keep_dir, 'submission_winning_blend.csv')
    final_submission.to_csv(keep_path, index=False)
    
    # Validate submission
    try:
        from validate_submission import validate_submission
        print("\nValidating winning ensemble...")
        result = validate_submission(output_path)
        if result and 'estimated_score' in result:
            print(f"Estimated score: {result['estimated_score']:.6f}")
            
            # Check if we've improved over individual models
            if result['estimated_score'] > 0.85:
                print("SUCCESS! The ensemble performance exceeds the benchmark!")
            else:
                # Try alternative weighting scheme 
                print("Trying an alternative weighting scheme...")
                ensemble_df['balanced_blend'] = ensemble_df.drop('id', axis=1).mean(axis=1)
                
                alt_submission = pd.DataFrame({
                    'id': ensemble_df['id'],
                    'rainfall': ensemble_df['balanced_blend']
                })
                
                alt_path = 'data/submission_balanced_blend.csv'
                alt_submission.to_csv(alt_path, index=False)
                
                alt_result = validate_submission(alt_path)
                if alt_result and alt_result['estimated_score'] > result['estimated_score']:
                    print(f"Balanced blend is better: {alt_result['estimated_score']:.6f}")
                    # Use the better version
                    final_submission = alt_submission
                    final_submission.to_csv(output_path, index=False)
                    final_submission.to_csv(keep_path, index=False)
    except Exception as e:
        print(f"Could not validate submission: {str(e)}")
    
    return final_submission

if __name__ == "__main__":
    create_winning_ensemble()
