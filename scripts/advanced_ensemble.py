import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.model_selection import KFold
import os
import glob

def create_super_ensemble():
    """Create a sophisticated ensemble from multiple base models"""
    print("Building super ensemble...")
    
    # Load all submission files from keep directory
    keep_dir = 'data/keep'
    submission_files = glob.glob(os.path.join(keep_dir, 'submission_*.csv'))
    
    if len(submission_files) < 3:
        print("Need at least 3 models for effective ensembling. Check 'data/keep' directory.")
        return None
    
    # Load submissions into a dictionary
    models = {}
    common_ids = None
    
    for file_path in submission_files:
        model_name = os.path.basename(file_path).replace('submission_', '').replace('.csv', '')
        df = pd.read_csv(file_path)
        
        if common_ids is None:
            common_ids = set(df['id'])
        else:
            common_ids = common_ids.intersection(set(df['id']))
        
        models[model_name] = df
    
    common_ids = sorted(list(common_ids))
    print(f"Found {len(models)} base models with {len(common_ids)} common IDs")
    
    # Create a dataframe for stacking
    stack_df = pd.DataFrame({'id': common_ids})
    
    # Add predictions from each model
    for name, df in models.items():
        df_filtered = df[df['id'].isin(common_ids)].sort_values('id')
        stack_df[name] = df_filtered['rainfall'].values
    
    # Try to load synthetic data for optimization
    try:
        synthetic_df = pd.read_csv('data/synthetic_bootstrap.csv')
        print(f"Loaded synthetic data with {len(synthetic_df)} samples")
        
        # Filter to common IDs
        common_synthetic_ids = set(synthetic_df['id']).intersection(set(stack_df['id']))
        if len(common_synthetic_ids) > 50:
            print(f"Using {len(common_synthetic_ids)} common IDs for optimization")
            
            # Prepare training data for meta-model
            train_ids = list(common_synthetic_ids)
            train_df = stack_df[stack_df['id'].isin(train_ids)].copy()
            
            # Get true values for training
            synthetic_subset = synthetic_df[synthetic_df['id'].isin(train_ids)]
            synthetic_subset = synthetic_subset.sort_values('id')
            
            # Normalize synthetic targets to 0-1 if needed
            true_rainfall = synthetic_subset['rainfall'].values
            if true_rainfall.max() > 1.5 or true_rainfall.min() < -0.5:
                print("Normalizing synthetic targets...")
                true_rainfall = (true_rainfall - true_rainfall.min()) / (true_rainfall.max() - true_rainfall.min())
            
            # Extract features and target
            X_train = train_df.drop('id', axis=1)
            y_train = true_rainfall
            
            # Train meta-models
            meta_models = {
                'bayesian': BayesianRidge(),
                'logistic': LogisticRegression(max_iter=1000, C=0.1)
            }
            
            for name, model in meta_models.items():
                model.fit(X_train, y_train)
                print(f"Trained {name} meta-model")
                
                # Generate predictions for all data
                X_all = stack_df.drop('id', axis=1)
                stack_df[f'meta_{name}'] = model.predict(X_all)
                
                # Save coefficients
                if hasattr(model, 'coef_'):
                    print(f"\n{name} model coefficients:")
                    for feat, coef in zip(X_train.columns, model.coef_.flatten()):
                        print(f"  {feat}: {coef:.6f}")
            
            # Create a final super ensemble using the meta-models + simple average
            stack_df['super_ensemble'] = stack_df[[f'meta_{name}' for name in meta_models.keys()]].mean(axis=1)
            
            # Also combine with simple average
            stack_df['final_blend'] = 0.7 * stack_df['super_ensemble'] + 0.3 * stack_df[list(models.keys())].mean(axis=1)
            
        else:
            print("Not enough common IDs with synthetic data for optimization")
            # Simple average if no optimization is possible
            stack_df['final_blend'] = stack_df[list(models.keys())].mean(axis=1)
    
    except Exception as e:
        print(f"Error using synthetic data: {str(e)}")
        # Simple average as fallback
        stack_df['final_blend'] = stack_df[list(models.keys())].mean(axis=1)
    
    # Create and save final submission
    final_submission = pd.DataFrame({
        'id': stack_df['id'],
        'rainfall': stack_df['final_blend']
    })
    
    output_path = 'data/submission_super_ensemble.csv'
    final_submission.to_csv(output_path, index=False)
    print(f"Super ensemble saved to {output_path}")
    
    # Copy to keep directory
    keep_path = os.path.join(keep_dir, 'submission_super_ensemble.csv')
    final_submission.to_csv(keep_path, index=False)
    print(f"Also saved to {keep_path}")
    
    # Validate the submission if possible
    try:
        from validate_submission import validate_submission
        result = validate_submission(output_path)
        if result and 'estimated_score' in result:
            print(f"Estimated score: {result['estimated_score']:.6f}")
    except:
        print("Could not validate submission")
    
    return final_submission

if __name__ == "__main__":
    create_super_ensemble()
