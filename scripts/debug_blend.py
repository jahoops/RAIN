import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def debug_winning_blend():
    """Debug why the winning blend values are so far off scale"""
    print("===== DEBUGGING WINNING BLEND CREATION =====\n")
    
    # Load the original submission file that has the problem
    orig_submission = pd.read_csv('data/submission_winning_blend.csv')
    print(f"Original submission range: {orig_submission['rainfall'].min():.4f} to {orig_submission['rainfall'].max():.4f}")
    print(f"Mean: {orig_submission['rainfall'].mean():.4f}, Std: {orig_submission['rainfall'].std():.4f}\n")
    
    # Let's check the keep folder for reference submissions
    keep_dir = os.path.join('data', 'keep')
    reference_files = [f for f in os.listdir(keep_dir) if f.startswith('submission_') and f.endswith('.csv')]
    
    for ref_file in reference_files:
        if ref_file == 'submission_winning_blend.csv':
            continue
            
        ref_df = pd.read_csv(os.path.join(keep_dir, ref_file))
        print(f"Reference {ref_file}: {ref_df['rainfall'].min():.4f} to {ref_df['rainfall'].max():.4f}, Mean: {ref_df['rainfall'].mean():.4f}")
    
    # Check for synthetic data used in optimization
    synthetic_files = [f for f in os.listdir('data') if 'synthetic' in f and f.endswith('.csv')]
    
    if synthetic_files:
        print("\n===== EXAMINING SYNTHETIC DATA =====")
        for syn_file in synthetic_files:
            syn_df = pd.read_csv(os.path.join('data', syn_file))
            if 'rainfall' in syn_df.columns:
                print(f"{syn_file}: {syn_df['rainfall'].min():.4f} to {syn_df['rainfall'].max():.4f}, Mean: {syn_df['rainfall'].mean():.4f}")
                # Sample some values
                print(f"Sample values: {syn_df['rainfall'].head(5).tolist()}")
    
    # Let's check the blend variations
    blend_files = [f for f in os.listdir('data') if f.startswith('submission_blend_') and f.endswith('.csv')]
    
    if blend_files:
        print("\n===== EXAMINING BLEND VARIATIONS =====")
        for blend_file in blend_files:
            blend_df = pd.read_csv(os.path.join('data', blend_file))
            print(f"{blend_file}: {blend_df['rainfall'].min():.4f} to {blend_df['rainfall'].max():.4f}, Mean: {blend_df['rainfall'].mean():.4f}")
    
    # Look at the problem more deeply - likely in the optimized blend
    print("\n===== ROOT CAUSE ANALYSIS =====")
    print("The most likely issue is in the 'optimized' blend variation, which uses Ridge regression")
    print("on synthetic data to create weights. Looking at synthetic_bootstrap.csv, it contains")
    print("actual rainfall values (around 1.0), not probabilistic predictions (0-1).")
    
    print("\nWhen the Ridge model is trained using these values directly, it learns to predict")
    print("raw rainfall values instead of probabilities, leading to predictions around 100+ mm.")
    
    print("\nThe fix is to normalize the synthetic targets before training, or to normalize")
    print("the final predictions to the expected range.")

    # Plot visualization of the issue
    try:
        plt.figure(figsize=(12, 6))
        
        # Plot 1: All submissions rainfall distributions  
        plt.subplot(1, 2, 1)
        plt.hist(orig_submission['rainfall'], bins=30, alpha=0.7, label='Problematic Blend')
        
        # Get a reference submission
        if reference_files:
            ref_df = pd.read_csv(os.path.join(keep_dir, reference_files[0]))
            plt.hist(ref_df['rainfall'], bins=30, alpha=0.7, label='Reference Submission')
        
        plt.title('Rainfall Distribution Comparison')
        plt.xlabel('Rainfall Values')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Plot 2: Correlation between blend and reference
        if reference_files:
            plt.subplot(1, 2, 2)
            
            # Merge data for correlation plot
            merged = pd.merge(orig_submission, ref_df, on='id', suffixes=('_prob', '_ref'))
            plt.scatter(merged['rainfall_ref'], merged['rainfall_prob'], alpha=0.5)
            plt.title('Correlation Between Submissions')
            plt.xlabel('Reference Submission Values')
            plt.ylabel('Problematic Blend Values')
            
        plt.tight_layout()
        plt.savefig('data/blend_debug_plot.png')
        print("\nSaved visualization to data/blend_debug_plot.png")
    except Exception as e:
        print(f"Could not create visualization: {str(e)}")

if __name__ == "__main__":
    debug_winning_blend()
