import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

def analyze_best_submission():
    """Analyze our best submission to understand its properties"""
    print("=== ANALYZING BEST SUBMISSION ===")
    
    # Load the best submission
    best_path = 'data/keep/submission_86403.csv'
    if not os.path.exists(best_path):
        print(f"ERROR: Best submission file not found at {best_path}")
        return None
    
    best_df = pd.read_csv(best_path)
    print(f"Loaded best submission with {len(best_df)} rows")
    
    # Analyze distribution
    print("\nStatistical properties:")
    print(f"Mean: {best_df['rainfall'].mean():.4f}")
    print(f"Median: {best_df['rainfall'].median():.4f}")
    print(f"Std Dev: {best_df['rainfall'].std():.4f}")
    print(f"Min: {best_df['rainfall'].min():.4f}")
    print(f"Max: {best_df['rainfall'].max():.4f}")
    print(f"Unique values: {len(best_df['rainfall'].unique())}")
    
    # Load other submissions for comparison
    keep_dir = 'data/keep'
    submission_files = glob.glob(os.path.join(keep_dir, 'submission_*.csv'))
    
    # Find which submissions are most similar to our best
    similarities = []
    
    for file_path in submission_files:
        if file_path == best_path:
            continue
            
        try:
            df = pd.read_csv(file_path)
            name = os.path.basename(file_path).replace('submission_', '').replace('.csv', '')
            
            # Find common IDs
            common_ids = set(best_df['id']).intersection(set(df['id']))
            if len(common_ids) > 0:
                # Filter to common IDs
                best_subset = best_df[best_df['id'].isin(common_ids)].sort_values('id')
                df_subset = df[df['id'].isin(common_ids)].sort_values('id')
                
                # Calculate correlation
                corr = np.corrcoef(best_subset['rainfall'], df_subset['rainfall'])[0, 1]
                
                # Calculate similarity metrics
                rmse = np.sqrt(np.mean((best_subset['rainfall'] - df_subset['rainfall'])**2))
                similarities.append({
                    'name': name,
                    'file': file_path,
                    'correlation': corr,
                    'rmse': rmse,
                    'common_ids': len(common_ids)
                })
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Sort by correlation
    similarities.sort(key=lambda x: x['correlation'], reverse=True)
    
    print("\nSubmissions most similar to best:")
    for i, sim in enumerate(similarities[:5]):
        print(f"{i+1}. {sim['name']}: correlation={sim['correlation']:.4f}, RMSE={sim['rmse']:.4f}")
    
    # Sort by dissimilarity (lowest correlation)
    similarities.sort(key=lambda x: x['correlation'])
    
    print("\nSubmissions most different from best:")
    for i, sim in enumerate(similarities[:5]):
        print(f"{i+1}. {sim['name']}: correlation={sim['correlation']:.4f}, RMSE={sim['rmse']:.4f}")
    
    # Create visualization directory
    os.makedirs('analysis/best_analysis', exist_ok=True)
    
    # Plot distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    sns.histplot(best_df['rainfall'], bins=30, kde=True)
    plt.title('Best Submission Distribution')
    plt.xlabel('Rainfall')
    
    plt.subplot(2, 2, 2)
    plt.scatter(best_df['id'], best_df['rainfall'], alpha=0.5, s=3)
    plt.title('Predictions by ID')
    plt.xlabel('ID')
    plt.ylabel('Rainfall')
    
    # Create correlation visualization with top similar and dissimilar
    top_similar = [s['name'] for s in similarities[-5:]]  # Most similar (reversed order)
    top_different = [s['name'] for s in similarities[:5]]  # Most different
    
    # Create a dataframe with the relevant predictions
    comparison_df = pd.DataFrame({'id': best_df['id'], 'best': best_df['rainfall']})
    
    for file_path in submission_files:
        name = os.path.basename(file_path).replace('submission_', '').replace('.csv', '')
        if name in top_similar or name in top_different:
            df = pd.read_csv(file_path)
            comparison_df = comparison_df.merge(
                df[['id', 'rainfall']].rename(columns={'rainfall': name}), 
                on='id', 
                how='left'
            )
    
    # Calculate correlation matrix
    corr_matrix = comparison_df.drop('id', axis=1).corr()
    
    plt.subplot(2, 2, 3)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Between Submissions')
    
    plt.subplot(2, 2, 4)
    plt.plot(rankdata(best_df['rainfall'])/len(best_df), label='Best (0.86403)')
    
    for name in top_similar[:2]:  # Plot just 2 most similar
        if name in comparison_df.columns:
            plt.plot(rankdata(comparison_df[name].fillna(0.5))/len(comparison_df), 
                    label=name, alpha=0.7)
    
    for name in top_different[:2]:  # Plot just 2 most different
        if name in comparison_df.columns:
            plt.plot(rankdata(comparison_df[name].fillna(0.5))/len(comparison_df), 
                    label=name, alpha=0.7, linestyle='--')
    
    plt.title('Cumulative Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('analysis/best_analysis/best_submission_analysis.png')
    print("\nSaved analysis visualization to analysis/best_analysis/best_submission_analysis.png")
    
    return best_df, similarities

def create_improved_ensemble(best_df, similarities):
    """Create an improved ensemble incorporating the best submission"""
    print("\n=== CREATING IMPROVED ENSEMBLE ===")
    
    # Select complementary submissions (high score but different from best)
    # We want both similar and different submissions for a diverse ensemble
    similar_models = [s for s in similarities[-7:]]  # Top similar
    different_models = [s for s in similarities[:7]]  # Top different
    
    # Filter to keep only those with decent correlation to best
    different_models = [s for s in different_models if s['correlation'] > 0]
    
    print(f"Using {len(similar_models)} similar models and {len(different_models)} complementary models")
    
    # Load all selected models
    ensemble_dfs = {}
    ensemble_dfs['best'] = best_df
    
    for model in similar_models + different_models:
        try:
            df = pd.read_csv(model['file'])
            ensemble_dfs[model['name']] = df
            print(f"Loaded {model['name']}")
        except Exception as e:
            print(f"Error loading {model['name']}: {str(e)}")
    
    # Find common IDs across all models
    common_ids = set(best_df['id'])
    for name, df in ensemble_dfs.items():
        common_ids = common_ids.intersection(set(df['id']))
    
    print(f"Found {len(common_ids)} IDs common to all selected models")
    
    if len(common_ids) < 10:
        print("WARNING: Very few common IDs. Using union of IDs instead.")
        # Use all IDs from best submission
        common_ids = set(best_df['id'])
    
    # Create a dataframe with all model predictions
    ensemble_df = pd.DataFrame({'id': sorted(list(common_ids))})
    
    # Add each model's predictions
    for name, df in ensemble_dfs.items():
        df_common = df[df['id'].isin(common_ids)].sort_values('id')
        ensemble_df[name] = df_common['rainfall'].values
    
    # Create simple weighted ensemble
    # Give high weight to best model, medium to similar models, low to different models
    best_weight = 0.5
    similar_weight = 0.3 / max(1, len(similar_models))
    different_weight = 0.2 / max(1, len(different_models))
    
    print(f"Ensemble weights: best={best_weight:.3f}, similar={similar_weight:.3f}/model, different={different_weight:.3f}/model")
    
    # Create weighted average
    ensemble_df['ensemble'] = best_weight * ensemble_df['best']
    
    for model in similar_models:
        name = model['name']
        if name in ensemble_df.columns:
            ensemble_df['ensemble'] += similar_weight * ensemble_df[name]
    
    for model in different_models:
        name = model['name']
        if name in ensemble_df.columns:
            ensemble_df['ensemble'] += different_weight * ensemble_df[name]
    
    # Create meta-model ensemble to potentially improve further
    try:
        print("\nTraining meta-ensemble model...")
        
        # Split IDs for cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        ensemble_df['meta'] = 0  # Column for meta-model predictions
        
        # Use all model predictions as features
        feature_cols = [name for name in ensemble_df.columns 
                      if name not in ['id', 'ensemble', 'meta']]
        
        # Use 'best' as target for meta-model (since it's our best model)
        for train_idx, val_idx in kf.split(ensemble_df):
            X_train = ensemble_df.iloc[train_idx][feature_cols]
            y_train = ensemble_df.iloc[train_idx]['best']
            X_val = ensemble_df.iloc[val_idx][feature_cols]
            
            model = GradientBoostingRegressor(
                n_estimators=100, 
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            ensemble_df.loc[ensemble_df.index[val_idx], 'meta'] = model.predict(X_val)
        
        # Train a final model on all data for any IDs not in the ensemble
        final_model = GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=42
        )
        
        final_model.fit(ensemble_df[feature_cols], ensemble_df['best'])
    except Exception as e:
        print(f"Error training meta-model: {str(e)}")
    
    # Create final ensemble - blend of weighted average and meta-model
    ensemble_df['final'] = 0.7 * ensemble_df['ensemble'] + 0.3 * ensemble_df['meta']
    
    # Save the improved ensemble
    improved_ensemble = pd.DataFrame({
        'id': ensemble_df['id'],
        'rainfall': ensemble_df['final']
    })
    
    # Handle any remaining IDs from the best submission
    missing_ids = set(best_df['id']) - set(improved_ensemble['id'])
    if missing_ids:
        print(f"Adding {len(missing_ids)} missing IDs from best submission")
        missing_df = best_df[best_df['id'].isin(missing_ids)]
        improved_ensemble = pd.concat([improved_ensemble, missing_df])
    
    # Ensure predictions are in valid range
    improved_ensemble['rainfall'] = np.clip(improved_ensemble['rainfall'], 0, 1)
    
    # Save ensemble
    os.makedirs('data/final', exist_ok=True)
    
    # Save as the final submission
    improved_path = 'data/final/submission_final.csv'
    improved_ensemble.to_csv(improved_path, index=False)
    print(f"Saved improved ensemble to {improved_path}")
    
    # Also save a copy with explicit name
    named_path = 'data/final/submission_improved_ensemble.csv'
    improved_ensemble.to_csv(named_path, index=False)
    print(f"Also saved as {named_path}")
    
    # Make a copy in keep directory
    keep_path = 'data/keep/submission_improved_ensemble.csv'
    improved_ensemble.to_csv(keep_path, index=False)
    
    # Validate the submission
    try:
        from fix_validation import validate_prediction_quality
        print("\nValidating improved ensemble:")
        result = validate_prediction_quality(improved_path)
        
        if result:
            quality_score = result['quality_score']
            print(f"\nQuality score: {quality_score}/90")
            print(f"Estimated score: {result['estimated_score']:.6f}")
            
            # If the quality score is good, we're done
            if quality_score >= 80:
                print("\nThe improved ensemble has excellent quality metrics!")
                print("Expected to perform better than the individual best model.")
                return improved_ensemble
    except Exception as e:
        print(f"Could not validate: {str(e)}")
    
    # Compare with best submission
    print("\nImproved ensemble distribution comparison:")
    print(f"Best mean: {best_df['rainfall'].mean():.4f} vs Ensemble mean: {improved_ensemble['rainfall'].mean():.4f}")
    print(f"Best std: {best_df['rainfall'].std():.4f} vs Ensemble std: {improved_ensemble['rainfall'].std():.4f}")
    
    return improved_ensemble

def apply_calibration(best_df):
    """Apply calibration techniques to potentially improve the submission"""
    print("\n=== APPLYING CALIBRATION TECHNIQUES ===")
    
    # Create a copy to work with
    calibrated_df = best_df.copy()
    
    # 1. Fine-tune the distribution
    rainfall = calibrated_df['rainfall'].values
    
    # Determine if the distribution is imbalanced
    mean_val = np.mean(rainfall)
    
    print(f"Current mean: {mean_val:.4f}")
    target_mean = 0.5  # Most balanced binary distribution
    
    if abs(mean_val - target_mean) > 0.05:
        print(f"Adjusting distribution to be more balanced (current mean: {mean_val:.4f})")
        
        # Apply a simple calibration to shift the mean closer to 0.5
        # while preserving the relative order of predictions
        
        # Linearly scale values to adjust mean
        alpha = target_mean / mean_val
        
        # This is a simple adjustment that maintains order but shifts the mean
        new_rainfall = rainfall * alpha
        
        # Ensure values stay in [0, 1] range
        new_rainfall = np.clip(new_rainfall, 0, 1)
        
        # Preserve extremes - ensure we still have some values close to 0 and 1
        if min(new_rainfall) > 0.05:
            lowest_5pct = np.percentile(new_rainfall, 5)
            new_rainfall[new_rainfall <= lowest_5pct] *= 0.5
        
        if max(new_rainfall) < 0.95:
            highest_5pct = np.percentile(new_rainfall, 95)
            new_rainfall[new_rainfall >= highest_5pct] = 1 - (1 - new_rainfall[new_rainfall >= highest_5pct]) * 0.5
        
        calibrated_df['rainfall'] = new_rainfall
        print(f"Adjusted mean: {np.mean(calibrated_df['rainfall']):.4f}")
    
    # 2. Try isotonic regression for another calibrated version
    try:
        from sklearn.isotonic import IsotonicRegression
        
        # We don't have ground truth, but we can calibrate based on the distribution
        # Assuming extreme prediction values (close to 0 or 1) are more likely to be correct
        
        # Create a synthetic target that amplifies the extremes
        synthetic_target = np.zeros_like(rainfall)
        synthetic_target[rainfall < 0.3] = 0
        synthetic_target[rainfall > 0.7] = 1
        synthetic_target[(rainfall >= 0.3) & (rainfall <= 0.7)] = rainfall[(rainfall >= 0.3) & (rainfall <= 0.7)]
        
        # Fit isotonic regression
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(rainfall, synthetic_target)
        
        # Create isotonic calibration
        isotonic_df = best_df.copy()
        isotonic_df['rainfall'] = iso_reg.predict(rainfall)
        
        # Ensure values stay in [0, 1] range
        isotonic_df['rainfall'] = np.clip(isotonic_df['rainfall'], 0, 1)
        
        print(f"Isotonic calibration - mean: {isotonic_df['rainfall'].mean():.4f}, std: {isotonic_df['rainfall'].std():.4f}")
    except Exception as e:
        print(f"Could not apply isotonic calibration: {str(e)}")
        isotonic_df = None
    
    # Create visualizations of calibrated distributions
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(best_df['rainfall'], bins=30, alpha=0.7, label='Original')
    sns.histplot(calibrated_df['rainfall'], bins=30, alpha=0.7, label='Calibrated')
    if isotonic_df is not None:
        sns.histplot(isotonic_df['rainfall'], bins=30, alpha=0.7, label='Isotonic')
    plt.title('Distribution Comparison')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(np.sort(best_df['rainfall']), label='Original')
    plt.plot(np.sort(calibrated_df['rainfall']), label='Calibrated')
    if isotonic_df is not None:
        plt.plot(np.sort(isotonic_df['rainfall']), label='Isotonic')
    plt.title('Sorted Values Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('analysis/best_analysis/calibration_comparison.png')
    print("Saved calibration visualization to analysis/best_analysis/calibration_comparison.png")
    
    # Save the calibrated versions
    calibrated_df.to_csv('data/keep/submission_calibrated.csv', index=False)
    print("Saved calibrated submission to data/keep/submission_calibrated.csv")
    
    if isotonic_df is not None:
        isotonic_df.to_csv('data/keep/submission_isotonic.csv', index=False)
        print("Saved isotonic calibration to data/keep/submission_isotonic.csv")
    
    return calibrated_df

if __name__ == "__main__":
    print("=== IMPROVING BEST SUBMISSION (Score: 0.86403) ===\n")
    
    # Create output directories
    os.makedirs('analysis/best_analysis', exist_ok=True)
    os.makedirs('data/keep', exist_ok=True)
    
    # Analyze the best submission
    best_df, similarities = analyze_best_submission()
    
    if best_df is not None:
        # Create improved ensemble
        improved_ensemble = create_improved_ensemble(best_df, similarities)
        
        # Apply calibration to best submission
        calibrated_df = apply_calibration(best_df)
        
        print("\n=== SUMMARY ===")
        print("Created the following improved submissions:")
        print("1. Improved Ensemble: data/final/submission_final.csv")
        print("2. Calibrated Best Model: data/keep/submission_calibrated.csv")
        
        print("\nRecommendation: Use the Improved Ensemble as your final submission.")
        print("It combines our best model with complementary models for potentially better performance.")
    else:
        print("Could not process best submission. Please check file paths.")
