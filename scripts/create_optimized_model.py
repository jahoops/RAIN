import pandas as pd
import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def find_best_base_model():
    """Find the best performing base model from existing submissions"""
    print("=== FINDING BEST BASE MODEL ===")
    
    # Locate synthetic data
    synthetic_path = None
    for path in ['data/synthetic_aligned.csv', 'data/synthetic_bootstrap.csv', 'data/synthetic_validation.csv']:
        if os.path.exists(path):
            synthetic_path = path
            break
    
    if not synthetic_path:
        print("No synthetic data found for evaluation")
        return None, None
        
    synthetic_df = pd.read_csv(synthetic_path)
    print(f"Loaded synthetic data from {synthetic_path} with {len(synthetic_df)} rows")
    
    # Load all submissions
    submission_files = []
    for location in ['data/keep', 'data']:
        submission_files.extend(glob.glob(os.path.join(location, 'submission_*.csv')))
    
    models = {}
    for file_path in submission_files:
        model_name = os.path.basename(file_path).replace('submission_', '').replace('.csv', '')
        try:
            df = pd.read_csv(file_path)
            models[model_name] = df
            print(f"Loaded {model_name} with {len(df)} predictions")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Find common IDs between each model and synthetic data
    scores = {}
    for name, df in models.items():
        common_ids = set(df['id']).intersection(set(synthetic_df['id']))
        if len(common_ids) < 50:
            print(f"Skipping {name}: only {len(common_ids)} common IDs")
            continue
            
        # Filter to common IDs
        model_subset = df[df['id'].isin(common_ids)].sort_values('id')
        synth_subset = synthetic_df[synthetic_df['id'].isin(common_ids)].sort_values('id')
        
        # Calculate AUC and correlation
        try:
            preds = model_subset['rainfall'].values
            true_vals = synth_subset['rainfall'].values
            
            auc = roc_auc_score(true_vals > 0.5, preds)
            corr = np.corrcoef(true_vals, preds)[0, 1]
            score = (auc + max(0, corr)) / 2
            
            scores[name] = {
                'auc': auc,
                'correlation': corr,
                'combined_score': score,
                'common_ids': len(common_ids)
            }
            
            print(f"{name}: AUC={auc:.4f}, Corr={corr:.4f}, Score={score:.4f}")
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
    
    if not scores:
        print("No models could be evaluated against synthetic data")
        return None, None
        
    # Find best model
    best_model = max(scores.items(), key=lambda x: x[1]['combined_score'])
    print(f"\nBest model: {best_model[0]} with score {best_model[1]['combined_score']:.4f}")
    
    best_file = None
    for file in submission_files:
        if best_model[0] in file:
            best_file = file
            break
    
    if best_file:
        best_df = pd.read_csv(best_file)
        print(f"Loaded best model from {best_file}")
        return best_model[0], best_df
    else:
        print("Could not load best model file")
        return None, None

def create_enhanced_model(base_model_name=None, base_model_df=None):
    """Create an enhanced model using the best base model as a foundation"""
    print("\n=== CREATING ENHANCED MODEL ===")
    
    # If no base model provided, find the best one
    if base_model_df is None:
        base_model_name, base_model_df = find_best_base_model()
        
    if base_model_df is None:
        print("No base model available - can't create enhanced model")
        return None
    
    # Try to find training data
    train_df = None
    for path in ['data/train_processed.csv', 'data/train.csv']:
        if os.path.exists(path):
            train_df = pd.read_csv(path)
            print(f"Loaded training data from {path}")
            break
    
    if train_df is None or 'rainfall' not in train_df.columns:
        print("Could not find training data with target variable")
        return base_model_df  # Return the base model if we can't enhance it
    
    # Try to find test data
    test_df = None
    for path in ['data/test_processed.csv', 'data/test.csv']:
        if os.path.exists(path):
            test_df = pd.read_csv(path)
            print(f"Loaded test data from {path}")
            break
    
    if test_df is None:
        print("Could not find test data")
        return base_model_df  # Return the base model if we can't enhance it
    
    # Check if IDs match the base model
    if not set(test_df['id']).issubset(set(base_model_df['id'])):
        print("WARNING: Test IDs don't match base model IDs")
        # Try to fix by mapping to common IDs
        common_ids = sorted(list(set(test_df['id']).intersection(set(base_model_df['id']))))
        if len(common_ids) < 0.9 * len(test_df):
            print(f"Only found {len(common_ids)} common IDs - not enough to continue")
            return base_model_df
    
    # Prepare features - use most basic numeric features to avoid issues
    print("Preparing features...")
    X_train = train_df.select_dtypes(include=['int64', 'float64'])
    X_test = test_df.select_dtypes(include=['int64', 'float64'])
    
    # Ensure we have the same columns
    common_cols = list(set(X_train.columns).intersection(set(X_test.columns)))
    common_cols = [c for c in common_cols if c not in ['id', 'rainfall']]
    
    if len(common_cols) < 3:
        print(f"Only found {len(common_cols)} usable features - not enough to train")
        return base_model_df
        
    print(f"Using {len(common_cols)} numeric features")
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    # Handle any remaining NaN values
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())
    
    y_train = train_df['rainfall'].copy()
    
    # Train a robust model with cross-validation
    print("Training enhanced model...")
    
    # Use target binarization for better stability
    y_binary = (y_train > 0.5).astype(int)
    
    # Create robust ensemble model
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
    
    # Train with cross-validation to assess stability
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in cv.split(X_train, y_binary):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_binary.iloc[train_idx], y_binary.iloc[val_idx]
        
        model.fit(X_cv_train, y_cv_train)
        preds = model.predict_proba(X_cv_val)[:, 1]
        score = roc_auc_score(y_cv_val, preds)
        cv_scores.append(score)
    
    print(f"Cross-validation AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Train final model on all data
    model.fit(X_train, y_binary)
    
    # Get feature importance
    importance = pd.DataFrame({
        'feature': common_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 features:")
    print(importance.head(10))
    
    # Create visualizations directory
    os.makedirs('analysis', exist_ok=True)
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=importance.head(20))
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('analysis/feature_importance.png')
    
    # Generate predictions
    test_preds = model.predict_proba(X_test)[:, 1]
    
    # Create submission with corrected IDs
    submission = pd.DataFrame({
        'id': test_df['id'],
        'rainfall': test_preds
    })
    
    # Create a blended model
    print("\nCreating blended model with base predictions...")
    
    # Merge with base model for all IDs
    base_submission = base_model_df.copy()
    
    # Merge new predictions with base predictions
    merged = base_submission.merge(submission, on='id', how='left', suffixes=('_base', ''))
    
    # For any IDs where we don't have new predictions, keep base predictions
    merged['rainfall'] = merged['rainfall'].fillna(merged['rainfall_base'])
    
    # Create blended predictions (70% new model, 30% base model)
    blend_ratio = 0.7
    merged['rainfall_blend'] = (blend_ratio * merged['rainfall'] + 
                              (1 - blend_ratio) * merged['rainfall_base'])
    
    # Create the final submission
    optimized_df = pd.DataFrame({
        'id': merged['id'],
        'rainfall': merged['rainfall_blend']
    })
    
    # Save all versions
    optimized_df.to_csv('data/submission_optimized.csv', index=False)
    submission.to_csv('data/submission_enhanced.csv', index=False)
    
    # Also save to keep directory
    os.makedirs('data/keep', exist_ok=True)
    optimized_df.to_csv('data/keep/submission_optimized.csv', index=False)
    submission.to_csv('data/keep/submission_enhanced.csv', index=False)
    
    print("\nSaved submissions:")
    print("- data/submission_optimized.csv (blended model)")
    print("- data/submission_enhanced.csv (new model only)")
    
    # Validate the optimized submission
    try:
        from validation_fixed import validate_submission
        print("\nValidating optimized submission:")
        validate_submission('data/submission_optimized.csv')
    except Exception as e:
        print(f"Could not validate: {e}")
    
    return optimized_df

if __name__ == "__main__":
    print("Creating optimized model...")
    optimized_df = create_enhanced_model()
    print("\nProcess complete!")
