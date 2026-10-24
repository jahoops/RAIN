import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def fixed_create_blend_variations(blend_df: pd.DataFrame, pred_columns: List[str], 
                           synthetic_df: pd.DataFrame = None) -> Dict[str, np.ndarray]:
    """Create various blend variations with proper scaling for synthetic data"""
    variations = {}
    
    # 1. Simple average blend
    variations['average'] = blend_df[pred_columns].mean(axis=1).values
    print(f"Average blend range: {variations['average'].min():.4f} to {variations['average'].max():.4f}")
    
    # 2. Weighted blend based on correlation matrix
    if len(pred_columns) >= 2:
        corr_matrix = blend_df[pred_columns].corr()
        # Higher correlation with other models = higher weight
        mean_corr = corr_matrix.mean()
        weights = mean_corr / mean_corr.sum()
        weighted_preds = np.zeros(len(blend_df))
        for col, weight in zip(pred_columns, weights):
            weighted_preds += blend_df[col].values * weight
        variations['correlation_weighted'] = weighted_preds
        print(f"Correlation weighted blend range: {weighted_preds.min():.4f} to {weighted_preds.max():.4f}")
    
    # 3. Rank averaging (more robust to outliers)
    if len(pred_columns) >= 2:
        ranks = blend_df[pred_columns].rank()
        variations['rank_average'] = ranks.mean(axis=1).values
        
        # Normalize rank average to 0-1 range for consistency
        min_val = variations['rank_average'].min()
        max_val = variations['rank_average'].max()
        variations['rank_average'] = (variations['rank_average'] - min_val) / (max_val - min_val)
        print(f"Rank average blend range: {variations['rank_average'].min():.4f} to {variations['rank_average'].max():.4f}")
    
    # 4. Optimized weighting using synthetic data - THIS IS WHERE THE BUG IS
    if synthetic_df is not None:
        common_ids = set(blend_df['id']).intersection(set(synthetic_df['id']))
        if len(common_ids) >= 50:  # Only if we have enough common samples
            # Merge with synthetic data to get ground truth
            train_df = blend_df[blend_df['id'].isin(common_ids)].copy()
            train_df = train_df.sort_values('id').reset_index(drop=True)
            
            synthetic_subset = synthetic_df[synthetic_df['id'].isin(common_ids)]
            synthetic_subset = synthetic_subset.sort_values('id').reset_index(drop=True)
            
            # *** DEBUG: Print synthetic data range before using it ***
            print(f"\nSynthetic data rainfall range: {synthetic_subset['rainfall'].min():.4f} to {synthetic_subset['rainfall'].max():.4f}")
            print(f"Sample synthetic values: {synthetic_subset['rainfall'].head(5).tolist()}")
            
            # *** FIX: Normalize the synthetic target to 0-1 range ***
            # Check if values need normalization (if they're not already 0-1 probabilities)
            if synthetic_subset['rainfall'].max() > 1.5 or synthetic_subset['rainfall'].min() < -0.5:
                print("WARNING: Synthetic data outside expected 0-1 range.")
                print("ORIGINAL ERROR: This is causing the Ridge model to predict raw values instead of probabilities!")
                print("Normalizing synthetic target values...")
                
                # Create a normalized copy for training
                normalized_rainfall = (synthetic_subset['rainfall'] - synthetic_subset['rainfall'].min()) / (synthetic_subset['rainfall'].max() - synthetic_subset['rainfall'].min())
                train_df['true_rainfall'] = normalized_rainfall.values
                print(f"Normalized target range: {train_df['true_rainfall'].min():.4f} to {train_df['true_rainfall'].max():.4f}")
            else:
                # Use original values if they're already in correct range
                train_df['true_rainfall'] = synthetic_subset['rainfall'].values
            
            # Train a Ridge model to learn optimal weights
            from sklearn.linear_model import RidgeCV
            model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], fit_intercept=False)
            model.fit(train_df[pred_columns], train_df['true_rainfall'])
            
            # Apply learned weights to full dataset
            optimized_preds = model.predict(blend_df[pred_columns])
            
            # *** ADDITIONAL CHECK: Ensure predictions are in expected range ***
            if optimized_preds.max() > 1.5 or optimized_preds.min() < -0.5:
                print("WARNING: Optimized predictions outside expected range. Normalizing...")
                optimized_preds = (optimized_preds - optimized_preds.min()) / (optimized_preds.max() - optimized_preds.min())
            
            variations['optimized'] = optimized_preds
            print(f"Optimized blend range: {optimized_preds.min():.4f} to {optimized_preds.max():.4f}")
            
            print("\nOptimized weights from synthetic data:")
            for col, coef in zip(pred_columns, model.coef_):
                print(f"{col}: {coef:.6f}")
    
    # 5. Min-max blend (experimental - averaging min and max predictions)
    min_preds = blend_df[pred_columns].min(axis=1).values
    max_preds = blend_df[pred_columns].max(axis=1).values
    variations['min_max'] = (min_preds + max_preds) / 2
    print(f"Min-max blend range: {variations['min_max'].min():.4f} to {variations['min_max'].max():.4f}")
    
    # Create submissions for each variation
    for name, preds in variations.items():
        sub_df = pd.DataFrame({'id': blend_df['id'], 'rainfall': preds})
        sub_df.to_csv(f'data/submission_blend_{name}.csv', index=False)
        print(f"Saved {name} blend: data/submission_blend_{name}.csv")
    
    return variations

def fixed_meta_ensemble(blend_variations: Dict[str, np.ndarray], ids: pd.Series) -> pd.DataFrame:
    """Create meta-ensemble with validation to prevent invalid values"""
    print("\nCreating meta-ensemble from all blend variations...")
    
    # Print the range for each variation before combining
    print("Variation ranges before ensemble:")
    for name, preds in blend_variations.items():
        print(f"{name}: {preds.min():.4f} to {preds.max():.4f}")
    
    # Combine all variation predictions with equal weight
    ensemble_preds = np.zeros(len(ids))
    for name, preds in blend_variations.items():
        ensemble_preds += preds / len(blend_variations)
    
    # CRITICAL CHECK: Validate final predictions are in expected range
    print(f"\nEnsemble predictions before validation: {ensemble_preds.min():.4f} to {ensemble_preds.max():.4f}")
    
    if ensemble_preds.max() > 1.5 or ensemble_preds.min() < -0.5:
        print("WARNING: Final predictions outside expected range (0-1)!")
        print("Normalizing to 0-1 range...")
        ensemble_preds = (ensemble_preds - ensemble_preds.min()) / (ensemble_preds.max() - ensemble_preds.min())
        print(f"After normalization: {ensemble_preds.min():.4f} to {ensemble_preds.max():.4f}")
    
    # In create_meta_ensemble function, add this before saving:
    if ensemble_preds.max() > 1.5 or ensemble_preds.min() < -0.5:
        print("WARNING: Final predictions outside expected range. Normalizing to 0-1 scale...")
        ensemble_preds = (ensemble_preds - ensemble_preds.min()) / (ensemble_preds.max() - ensemble_preds.min())

    # Create final submission
    final_blend = pd.DataFrame({'id': ids, 'rainfall': ensemble_preds})
    
    # Save fixed submission
    final_path = 'data/submission_winning_blend_fixed.csv'
    final_blend.to_csv(final_path, index=False)
    print(f"Saved fixed winning blend: {final_path}")
    
    # Copy to keep directory
    import shutil
    keep_path = os.path.join('data', 'keep', 'submission_winning_blend_fixed.csv')
    shutil.copy2(final_path, keep_path)
    print(f"Saved copy to {keep_path}")
    
    return final_blend

def run_debug_and_fix():
    """Run the debugging and apply the fix"""
    # First, debug to understand the issue
    print("========= STEP 1: DEBUG THE ISSUE =========")
    from debug_blend import debug_winning_blend
    debug_winning_blend()
    
    print("\n\n========= STEP 2: APPLY THE FIX =========")
    print("The fix uses a modified blend creation function with proper validation")
    print("and normalization of the synthetic data, which was the root cause.")
    
    print("\nTo apply this fix to your winning_blend.py file, replace the")
    print("create_blend_variations function with the fixed version, and")
    print("add validation checks to create_meta_ensemble.")
    
    print("\nThe fix ensures that the Ridge model is trained on normalized targets")
    print("and that the final ensemble predictions are always in the expected 0-1 range.")

if __name__ == "__main__":
    run_debug_and_fix()
