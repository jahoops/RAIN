import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

def analyze_predictions():
    """Analyze and compare different model predictions"""
    print("Analyzing model predictions...")
    
    # Find all submissions
    keep_dir = 'data/keep'
    submission_files = glob.glob(os.path.join(keep_dir, 'submission_*.csv'))
    
    if not submission_files:
        print("No submission files found in keep directory")
        return
    
    # Load submissions
    submissions = {}
    for file_path in submission_files:
        model_name = os.path.basename(file_path).replace('submission_', '').replace('.csv', '')
        try:
            submissions[model_name] = pd.read_csv(file_path)
        except:
            print(f"Could not load {file_path}")
    
    print(f"Loaded {len(submissions)} submission files")
    
    # Create directory for plots
    os.makedirs('analysis', exist_ok=True)
    
    # 1. Compare prediction distributions
    plt.figure(figsize=(12, 8))
    
    for i, (name, df) in enumerate(submissions.items()):
        if i >= 6:  # Limit to 6 models for clarity
            break
        sns.kdeplot(df['rainfall'], label=name, fill=True, alpha=0.3)
    
    plt.title('Prediction Distributions Across Models')
    plt.xlabel('Rainfall Prediction')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('analysis/prediction_distributions.png')
    print("Saved prediction distributions plot")
    
    # 2. Compare histogram of predictions
    plt.figure(figsize=(12, 8))
    
    for i, (name, df) in enumerate(submissions.items()):
        if i >= 6:  # Limit to 6 models for clarity
            break
        plt.hist(df['rainfall'], bins=20, alpha=0.3, label=name)
    
    plt.title('Histogram of Predictions')
    plt.xlabel('Rainfall Prediction')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('analysis/prediction_histograms.png')
    
    # 3. Create correlation matrix between predictions
    # Find submissions with common IDs
    model_names = list(submissions.keys())
    first_model = model_names[0]
    common_ids = set(submissions[first_model]['id'])
    
    for name in model_names[1:]:
        common_ids = common_ids.intersection(set(submissions[name]['id']))
    
    common_ids = sorted(list(common_ids))
    
    if not common_ids:
        print("No common IDs across submissions")
    else:
        print(f"Found {len(common_ids)} common IDs across all submissions")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({'id': common_ids})
        
        for name, df in submissions.items():
            subset = df[df['id'].isin(common_ids)].sort_values('id')
            comparison_df[name] = subset['rainfall'].values
        
        # Calculate correlation matrix
        corr = comparison_df.drop('id', axis=1).corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation Between Model Predictions')
        plt.savefig('analysis/prediction_correlations.png')
        print("Saved correlation matrix")
        
        # 4. Look for patterns where models disagree
        plt.figure(figsize=(12, 10))
        
        # Find most different model pairs
        model_cols = comparison_df.columns[1:]
        max_diff = 0
        max_diff_pair = None
        
        for i, model1 in enumerate(model_cols):
            for model2 in model_cols[i+1:]:
                diff = np.abs(comparison_df[model1] - comparison_df[model2]).mean()
                if diff > max_diff:
                    max_diff = diff
                    max_diff_pair = (model1, model2)
        
        if max_diff_pair:
            print(f"Models with biggest differences: {max_diff_pair[0]} vs {max_diff_pair[1]} (avg diff: {max_diff:.4f})")
            plt.scatter(comparison_df[max_diff_pair[0]], comparison_df[max_diff_pair[1]], alpha=0.3)
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlabel(max_diff_pair[0])
            plt.ylabel(max_diff_pair[1])
            plt.title(f'Comparison of {max_diff_pair[0]} vs {max_diff_pair[1]}')
            plt.savefig('analysis/model_differences.png')
        
        # 5. Generate ensemble suggestions based on correlation
        print("\nEnsemble suggestions based on correlation analysis:")
        
        # Models with lower correlation are better for ensembling
        def suggest_ensemble(corr_matrix, top_n=3):
            # Get average correlation for each model
            avg_corr = corr_matrix.mean() - 1.0  # Subtract self-correlation 
            
            # Sort by lowest correlation (most diverse)
            diverse_models = avg_corr.sort_values().index[:top_n]
            print(f"Most diverse models: {', '.join(diverse_models)}")
            
            # Sort by highest correlation (most accurate if we assume correlation = accuracy)
            accurate_models = avg_corr.sort_values(ascending=False).index[:top_n]
            print(f"Most similar models: {', '.join(accurate_models)}")
            
            # Suggest mix of both
            suggested = list(set(list(diverse_models) + list(accurate_models)))
            print(f"Suggested ensemble: {', '.join(suggested)}")
            
            # Recommend weights
            weights = {}
            
            # More weight to models that agree with others (higher correlation)
            total = sum(avg_corr[suggested] + 1)  # Shift to positive values
            for model in suggested:
                weights[model] = (avg_corr[model] + 1) / total
            
            return weights
        
        suggested_weights = suggest_ensemble(corr)
        print("\nSuggested ensemble weights:")
        for model, weight in suggested_weights.items():
            print(f"{model}: {weight:.4f}")
        
        # 6. Create suggested ensemble
        suggested_blend = np.zeros(len(comparison_df))
        
        for model, weight in suggested_weights.items():
            suggested_blend += comparison_df[model].values * weight
        
        comparison_df['suggested_blend'] = suggested_blend
        
        # Create equal-weight ensemble for comparison
        comparison_df['equal_blend'] = comparison_df[model_cols].mean(axis=1)
        
        # Save suggested ensemble
        suggested_df = pd.DataFrame({
            'id': comparison_df['id'],
            'rainfall': comparison_df['suggested_blend']
        })
        
        suggested_path = 'data/submission_suggested_ensemble.csv'
        suggested_df.to_csv(suggested_path, index=False)
        
        equal_weight_df = pd.DataFrame({
            'id': comparison_df['id'],
            'rainfall': comparison_df['equal_blend']
        })
        
        equal_path = 'data/submission_equal_ensemble.csv'
        equal_weight_df.to_csv(equal_path, index=False)
        
        print(f"\nSaved suggested ensemble to {suggested_path}")
        print(f"Saved equal-weight ensemble to {equal_path}")
        
        # 7. Validate both ensembles
        try:
            from validate_submission import validate_submission
            print("\nValidating ensembles...")
            
            suggested_result = validate_submission(suggested_path)
            print(f"Suggested ensemble score: {suggested_result['estimated_score']:.6f}")
            
            equal_result = validate_submission(equal_path)
            print(f"Equal-weight ensemble score: {equal_result['estimated_score']:.6f}")
        except Exception as e:
            print(f"Could not validate ensembles: {str(e)}")

if __name__ == "__main__":
    analyze_predictions()
