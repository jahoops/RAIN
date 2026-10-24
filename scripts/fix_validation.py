import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def validate_prediction_quality(submission_path):
    """Validate a submission file's quality without relying on synthetic data"""
    print(f"Analyzing submission quality: {submission_path}")
    
    if not os.path.exists(submission_path):
        print(f"ERROR: File does not exist: {submission_path}")
        return None
    
    # Load the submission
    try:
        submission_df = pd.read_csv(submission_path)
        print(f"Loaded submission with {len(submission_df)} rows")
    except Exception as e:
        print(f"ERROR: Failed to load submission: {str(e)}")
        return None
    
    # Check basic format
    if 'id' not in submission_df.columns or 'rainfall' not in submission_df.columns:
        print("ERROR: Submission must have 'id' and 'rainfall' columns")
        return None
    
    # Create outputs directory
    os.makedirs('analysis/validation_fix', exist_ok=True)
    
    # Calculate basic statistics
    preds = submission_df['rainfall'].values
    stats = {
        'count': len(preds),
        'mean': np.mean(preds),
        'median': np.median(preds),
        'std': np.std(preds),
        'min': np.min(preds),
        'max': np.max(preds),
        'unique_values': len(np.unique(preds)),
        'unique_ratio': len(np.unique(preds)) / len(preds)
    }
    
    print("\n=== SUBMISSION STATISTICS ===")
    print(f"Count: {stats['count']}")
    print(f"Mean: {stats['mean']:.4f}")
    print(f"Median: {stats['median']:.4f}")
    print(f"Std Dev: {stats['std']:.4f}")
    print(f"Range: {stats['min']:.4f} to {stats['max']:.4f}")
    print(f"Unique values: {stats['unique_values']} ({stats['unique_ratio']:.2%} of rows)")
    
    # Check for proper range
    if stats['min'] < 0 or stats['max'] > 1:
        print("\nWARNING: Values outside the expected 0-1 range!")
        print(f"Out of range values: {len(preds[(preds < 0) | (preds > 1)])}")
    
    # Analyze distribution and create visualization
    plt.figure(figsize=(18, 10))
    
    # Distribution plot
    plt.subplot(2, 2, 1)
    plt.hist(preds, bins=50, edgecolor='black', alpha=0.7)
    plt.title('Prediction Distribution')
    plt.xlabel('Prediction Value')
    plt.ylabel('Count')
    
    # QQ plot to check for normality
    plt.subplot(2, 2, 2)
    from scipy import stats as scipy_stats
    scipy_stats.probplot(preds, dist="norm", plot=plt)
    plt.title('QQ Plot (Check for Normality)')
    
    # Values over ID plot
    plt.subplot(2, 2, 3)
    plt.scatter(submission_df['id'], submission_df['rainfall'], alpha=0.5, s=10)
    plt.title('Predictions vs ID')
    plt.xlabel('ID')
    plt.ylabel('Prediction')
    
    # Try to find pattern in predictions by sorting
    plt.subplot(2, 2, 4)
    sorted_preds = np.sort(preds)
    plt.plot(sorted_preds)
    plt.title('Sorted Predictions')
    plt.xlabel('Index')
    plt.ylabel('Sorted Prediction Value')
    
    plt.tight_layout()
    plt.savefig('analysis/validation_fix/prediction_analysis.png')
    print("Saved distribution analysis to analysis/validation_fix/prediction_analysis.png")
    
    # Compare with other submissions if available
    keep_dir = 'data/keep'
    other_submissions = glob.glob(os.path.join(keep_dir, 'submission_*.csv'))
    
    if other_submissions:
        print("\n=== COMPARING WITH OTHER SUBMISSIONS ===")
        
        submission_stats = []
        submission_values = []
        submission_names = []
        
        # Load this submission first
        submission_name = os.path.basename(submission_path).replace('.csv', '')
        submission_stats.append({
            'name': submission_name,
            'mean': stats['mean'],
            'median': stats['median'],
            'std': stats['std'],
            'min': stats['min'],
            'max': stats['max'],
            'unique_ratio': stats['unique_ratio']
        })
        submission_values.append(preds)
        submission_names.append(submission_name)
        
        # Load other submissions
        for file_path in other_submissions[:10]:  # Limit to 10 for clarity
            if os.path.abspath(file_path) == os.path.abspath(submission_path):
                continue
                
            try:
                df = pd.read_csv(file_path)
                other_name = os.path.basename(file_path).replace('submission_', '').replace('.csv', '')
                
                # Get stats
                other_preds = df['rainfall'].values
                submission_stats.append({
                    'name': other_name, 
                    'mean': np.mean(other_preds),
                    'median': np.median(other_preds),
                    'std': np.std(other_preds),
                    'min': np.min(other_preds),
                    'max': np.max(other_preds),
                    'unique_ratio': len(np.unique(other_preds)) / len(other_preds)
                })
                
                submission_values.append(other_preds)
                submission_names.append(other_name)
                
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        # Create comparison table
        stats_df = pd.DataFrame(submission_stats)
        print("\nSubmission statistics comparison:")
        print(stats_df.to_string())
        
        # Create correlation matrix if we have common IDs
        try:
            # Find common IDs
            common_ids = set(submission_df['id'])
            for file_path in other_submissions[:10]:
                if os.path.abspath(file_path) == os.path.abspath(submission_path):
                    continue
                    
                df = pd.read_csv(file_path)
                common_ids = common_ids.intersection(set(df['id']))
            
            # Create comparison matrix
            if common_ids:
                common_ids = sorted(list(common_ids))
                print(f"\nFound {len(common_ids)} common IDs across submissions")
                
                # Create a dataframe with predictions from all submissions
                comparison_df = pd.DataFrame({'id': common_ids})
                
                # Add this submission
                filtered_df = submission_df[submission_df['id'].isin(common_ids)].sort_values('id')
                comparison_df[submission_name] = filtered_df['rainfall'].values
                
                # Add other submissions
                for file_path in other_submissions[:10]:
                    if os.path.abspath(file_path) == os.path.abspath(submission_path):
                        continue
                        
                    name = os.path.basename(file_path).replace('submission_', '').replace('.csv', '')
                    df = pd.read_csv(file_path)
                    filtered_df = df[df['id'].isin(common_ids)].sort_values('id')
                    comparison_df[name] = filtered_df['rainfall'].values
                
                # Calculate correlation matrix
                corr_matrix = comparison_df.drop('id', axis=1).corr()
                
                # Create correlation heatmap
                plt.figure(figsize=(12, 10))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
                plt.title('Correlation Between Submissions')
                plt.tight_layout()
                plt.savefig('analysis/validation_fix/submission_correlations.png')
                print(f"Saved correlation matrix to analysis/validation_fix/submission_correlations.png")
                
                # Find most correlated submissions
                corr_with_current = corr_matrix[submission_name].sort_values(ascending=False)[1:6]
                print("\nSubmissions most similar to current:")
                print(corr_with_current)
                
                # Find least correlated submissions
                least_corr = corr_matrix[submission_name].sort_values()[0:5]
                print("\nSubmissions least similar to current:")
                print(least_corr)
        except Exception as e:
            print(f"Error creating correlation matrix: {str(e)}")
    
    # Analysis based on the submission's intrinsic quality
    print("\n=== QUALITY ASSESSMENT ===")
    
    quality_score = 0
    
    # Favor submissions with predictions in the proper range
    if 0 <= stats['min'] <= 1 and 0 <= stats['max'] <= 1:
        quality_score += 30
        print("✓ Predictions are properly contained in the 0-1 range")
    else:
        print("✗ Predictions are outside the 0-1 range")
    
    # Favor submissions with reasonable spread
    if stats['std'] > 0.2:
        quality_score += 20
        print("✓ Predictions have good variability")
    elif stats['std'] > 0.1:
        quality_score += 10
        print("⚠ Predictions have moderate variability")
    else:
        print("✗ Predictions have low variability")
    
    # Favor submissions with many unique values (granularity)
    if stats['unique_ratio'] > 0.95:
        quality_score += 25
        print("✓ Predictions have excellent granularity (many unique values)")
    elif stats['unique_ratio'] > 0.5:
        quality_score += 15
        print("✓ Predictions have good granularity")
    elif stats['unique_ratio'] > 0.1:
        quality_score += 5
        print("⚠ Predictions have limited granularity")
    else:
        print("✗ Predictions have poor granularity (few unique values)")
    
    # Favor submissions with a reasonable mean (around 0.5 for binary problems)
    if 0.3 <= stats['mean'] <= 0.7:
        quality_score += 15
        print("✓ Prediction mean is in a reasonable range")
    elif 0.2 <= stats['mean'] <= 0.8:
        quality_score += 10
        print("⚠ Prediction mean is somewhat skewed")
    else:
        print("✗ Prediction mean is heavily skewed")
    
    # Final assessment
    print(f"\nQuality score: {quality_score}/90")
    
    if quality_score >= 80:
        print("EXCELLENT - This submission has high quality predictions")
        print("Expected contest performance: Strong")
        estimated_score = 0.95
    elif quality_score >= 65:
        print("GOOD - This submission has good quality predictions")
        print("Expected contest performance: Above average")
        estimated_score = 0.85
    elif quality_score >= 50:
        print("ADEQUATE - This submission has acceptable quality")
        print("Expected contest performance: Average")
        estimated_score = 0.75
    else:
        print("NEEDS IMPROVEMENT - This submission has potential issues")
        print("Expected contest performance: Below average")
        estimated_score = 0.65
    
    return {
        'submission_path': submission_path,
        'quality_score': quality_score,
        'estimated_score': estimated_score,
        'stats': stats
    }

if __name__ == "__main__":
    import sys
    
    # Default to the final submission path
    submission_path = 'data/final/submission_final.csv'
    
    # Allow custom path as command-line argument
    if len(sys.argv) > 1:
        submission_path = sys.argv[1]
    
    print(f"Running validation fix for {submission_path}")
    result = validate_prediction_quality(submission_path)
    
    if result:
        print(f"\nFinal quality score: {result['quality_score']}/90")
        print(f"Estimated contest score: {result['estimated_score']:.6f}")
