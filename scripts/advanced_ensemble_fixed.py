import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.model_selection import KFold
import os
import glob

def create_super_ensemble():
    """Create a sophisticated ensemble from multiple base models with better ID handling"""
    print("Building super ensemble with improved ID handling...")
    
    # Load all submission files from keep directory
    keep_dir = 'data/keep'
    submission_files = glob.glob(os.path.join(keep_dir, 'submission_*.csv'))
    
    if len(submission_files) < 3:
        print("Need at least 3 models for effective ensembling. Check 'data/keep' directory.")
        return None
    
    # Load submissions into a dictionary and track ID counts
    models = {}
    id_counts = {}
    all_ids = set()
    
    print("Loading submission files and analyzing IDs...")
    for file_path in submission_files:
        model_name = os.path.basename(file_path).replace('submission_', '').replace('.csv', '')
        try:
            df = pd.read_csv(file_path)
            if 'id' not in df.columns or 'rainfall' not in df.columns:
                print(f"Skipping {file_path} - missing required columns")
                continue
                
            # Check for proper format
            print(f"Loaded {model_name}: {len(df)} rows, rainfall range: {df['rainfall'].min():.4f}-{df['rainfall'].max():.4f}")
            
            # Track each ID and how many models have it
            for id_val in df['id']:
                if id_val not in id_counts:
                    id_counts[id_val] = 0
                id_counts[id_val] += 1
                all_ids.add(id_val)
                
            models[model_name] = df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if len(models) < 3:
        print("Not enough valid models found for ensembling.")
        return None
    
    # Find IDs that are present in at least 75% of models
    min_models = max(3, int(len(models) * 0.75))
    common_ids = [id_val for id_val, count in id_counts.items() if count >= min_models]
    
    if not common_ids:
        print("WARNING: No common IDs found across 75% of models. Falling back to IDs present in at least 3 models.")
        common_ids = [id_val for id_val, count in id_counts.items() if count >= 3]
    
    common_ids = sorted(common_ids)
    print(f"Using {len(common_ids)} IDs that appear in at least {min_models} of {len(models)} models")
    
    if len(common_ids) == 0:
        print("ERROR: No common IDs found. Cannot create ensemble.")
        # Print ID statistics to help diagnose the issue
        id_stats = pd.Series(id_counts).value_counts().sort_index()
        print("ID occurrence counts:")
        print(id_stats)
        
        # Check if we have any test file to get the correct IDs
        try:
            test_file = os.path.join('data', 'test.csv')
            if os.path.exists(test_file):
                test_df = pd.read_csv(test_file)
                print(f"Found test file with {len(test_df)} IDs")
                # Take the first submission and reindex it to match the test IDs
                if models:
                    first_model = list(models.values())[0]
                    submission = pd.DataFrame({'id': test_df['id']})
                    submission = submission.merge(first_model, on='id', how='left')
                    submission['rainfall'] = submission['rainfall'].fillna(0.5)  # Default value for missing
                    
                    submission = submission[['id', 'rainfall']]
                    output_path = 'data/submission_fallback.csv'
                    submission.to_csv(output_path, index=False)
                    print(f"Created fallback submission using first model: {output_path}")
                    return submission
            return None
        except Exception as e:
            print(f"Error creating fallback submission: {e}")
            return None
    
    # Create a dataframe for stacking
    stack_df = pd.DataFrame({'id': common_ids})
    
    # Add predictions from each model, handling missing IDs
    for name, df in models.items():
        # Only use models that have a good portion of the common IDs
        common_in_model = len(set(df['id']).intersection(common_ids))
        coverage = common_in_model / len(common_ids)
        
        if coverage < 0.5:  # Skip models with poor coverage
            print(f"Skipping {name} - only covers {coverage:.1%} of common IDs")
            continue
            
        # Merge to add this model's predictions
        model_df = df[df['id'].isin(common_ids)].copy()
        model_df = model_df.sort_values('id')
        
        # Check if we need to pad with missing values
        if len(model_df) < len(common_ids):
            print(f"Padding {name} with {len(common_ids) - len(model_df)} missing values")
            stack_df = stack_df.merge(model_df[['id', 'rainfall']], on='id', how='left')
            stack_df.rename(columns={'rainfall': name}, inplace=True)
            # Fill missing values with the mean
            stack_df[name].fillna(model_df['rainfall'].mean(), inplace=True)
        else:
            stack_df[name] = model_df['rainfall'].values
    
    # Drop any models that weren't added
    model_cols = [col for col in stack_df.columns if col != 'id']
    print(f"Ensemble will use {len(model_cols)} models")
    
    if len(model_cols) < 3:
        print("Not enough models with common IDs. Cannot create reliable ensemble.")
        return None
    
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
                print(f"Normalizing synthetic targets from range {true_rainfall.min():.4f}-{true_rainfall.max():.4f}")
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
            stack_df['final_blend'] = 0.7 * stack_df['super_ensemble'] + 0.3 * stack_df[model_cols].mean(axis=1)
            
        else:
            print("Not enough common IDs with synthetic data for optimization")
            # Simple average if no optimization is possible
            stack_df['final_blend'] = stack_df[model_cols].mean(axis=1)
    
    except Exception as e:
        print(f"Error using synthetic data: {str(e)}")
        # Simple average as fallback
        stack_df['final_blend'] = stack_df[model_cols].mean(axis=1)
    
    # Create and save final submission
    final_submission = pd.DataFrame({
        'id': stack_df['id'],
        'rainfall': stack_df['final_blend']
    })
    
    # Check for any NaN values
    if final_submission['rainfall'].isna().any():
        print("WARNING: NaN values in final predictions. Replacing with 0.5.")
        final_submission['rainfall'] = final_submission['rainfall'].fillna(0.5)
    
    output_path = 'data/submission_super_ensemble_fixed.csv'
    final_submission.to_csv(output_path, index=False)
    print(f"Super ensemble saved to {output_path}")
    
    # Copy to keep directory
    keep_path = os.path.join(keep_dir, 'submission_super_ensemble_fixed.csv')
    final_submission.to_csv(keep_path, index=False)
    print(f"Also saved to {keep_path}")
    
    # Validate the submission if possible
    try:
        from validate_submission import validate_submission
        result = validate_submission(output_path)
        if result and 'estimated_score' in result:
            print(f"Estimated score: {result['estimated_score']:.6f}")
    except Exception as e:
        print(f"Could not validate submission: {str(e)}")
    
    return final_submission

if __name__ == "__main__":
    print("Running fixed advanced ensemble...")
    create_super_ensemble()
