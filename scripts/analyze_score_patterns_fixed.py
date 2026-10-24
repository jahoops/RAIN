import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def extract_score_from_filename(filename):
    # Split the filename by the dot character
    parts = filename.split('.')
    # Extract the number from the middle part
    number_str = parts[1]
    # Convert to a number to the right of the decimal point
    number_decimal = float('0.' + number_str)
    return number_decimal

def analyze_submission_scores():
    """Analyze what patterns differentiate high-scoring from low-scoring submissions"""
    print("=== ANALYZING SCORE PATTERNS ===")
    
    # Check for actual submissions directory
    actual_dir = '/home/USER/Projects/RAIN/data/Actual'
    if not os.path.exists(actual_dir):
        print(f"ERROR: Actual submissions directory not found at {actual_dir}")
        return
        
    # Find all CSV files that match the naming pattern
    files = glob.glob(os.path.join(actual_dir, '*.csv'))
    if not files:
        print(f"No CSV files found in {actual_dir}")
        return
        
    # Load submissions and extract scores
    submissions = []
    
    for file_path in files:
        try:
            filename = os.path.basename(file_path)
            score = extract_score_from_filename(filename)
            df = pd.read_csv(file_path)
            
            if 'id' in df.columns and 'rainfall' in df.columns:
                submissions.append({
                    'filename': filename,
                    'score': score,
                    'df': df
                })
                print(f"Loaded {filename} with score {score:.6f} ({len(df)} rows)")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if not submissions:
        print("No valid submissions found with scores")
        return
    
    # Sort by score (highest first)
    submissions.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\nAnalyzing {len(submissions)} submissions")
    print(f"Scores range: {submissions[-1]['score']:.6f} to {submissions[0]['score']:.6f}")
    
    # Create directory for analysis
    os.makedirs('analysis/scores', exist_ok=True)
    
    # Calculate statistical features for each submission
    features = []
    
    for sub in submissions:
        rainfall = sub['df']['rainfall'].values
        
        # Handle any potential NaN values
        rainfall = rainfall[~np.isnan(rainfall)]
        
        # Calculate statistics
        stats_dict = {
            'score': sub['score'],
            'mean': np.mean(rainfall),
            'median': np.median(rainfall),
            'std': np.std(rainfall),
            'min': np.min(rainfall),
            'max': np.max(rainfall),
            'range': np.max(rainfall) - np.min(rainfall),
            'q25': np.percentile(rainfall, 25),
            'q75': np.percentile(rainfall, 75),
            'iqr': np.percentile(rainfall, 75) - np.percentile(rainfall, 25),
            'skew': stats.skew(rainfall),
            'kurtosis': stats.kurtosis(rainfall),
            'unique_ratio': len(np.unique(rainfall)) / len(rainfall),
            'zeros': np.sum(rainfall == 0) / len(rainfall),
            'ones': np.sum(rainfall == 1) / len(rainfall),
            'out_of_range': np.sum((rainfall < 0) | (rainfall > 1)) / len(rainfall),
            'below_0.1': np.sum(rainfall < 0.1) / len(rainfall),
            'above_0.9': np.sum(rainfall > 0.9) / len(rainfall),
            'mid_range': np.sum((rainfall >= 0.4) & (rainfall <= 0.6)) / len(rainfall)
        }
        
        # Store filename separately to avoid correlation issues
        stats_dict['filename'] = sub['filename']
        
        features.append(stats_dict)
    
    # Convert to DataFrame for easier analysis
    features_df = pd.DataFrame(features)
    
    # Calculate correlations with score - excluding non-numeric columns
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    score_correlations = features_df[numeric_cols].corr()['score'].drop('score').sort_values(ascending=False)
    
    print("\n=== CORRELATIONS WITH SCORE ===")
    for feature, corr in score_correlations.items():
        print(f"{feature:<15}: {corr:+.4f}")
    
    # Save correlations to CSV
    corr_df = pd.DataFrame({'feature': score_correlations.index, 'correlation': score_correlations.values})
    corr_df.to_csv('analysis/scores/feature_correlations.csv', index=False)
    print("Saved correlations to analysis/scores/feature_correlations.csv")
    
    # Create visualizations
    
    # 1. Correlation heatmap
    plt.figure(figsize=(14, 12))
    corr_matrix = features_df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Feature Correlations with Score')
    plt.tight_layout()
    plt.savefig('analysis/scores/correlation_heatmap.png')
    
    # 2. Top 6 correlating features with score
    plt.figure(figsize=(18, 15))
    top_features = score_correlations.abs().sort_values(ascending=False).head(6).index
    
    for i, feature in enumerate(top_features):
        plt.subplot(2, 3, i+1)
        plt.scatter(features_df['score'], features_df[feature])
        
        # Add trendline
        z = np.polyfit(features_df['score'], features_df[feature], 1)
        p = np.poly1d(z)
        plt.plot(features_df['score'], p(features_df['score']), "r--", alpha=0.8)
        
        plt.title(f'Score vs {feature} (r={score_correlations[feature]:.4f})')
        plt.xlabel('Score')
        plt.ylabel(feature)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis/scores/top_correlations.png')
    
    # 3. Distribution comparison: High vs Low scoring submissions
    if len(submissions) >= 2:
        # Split into high and low scoring (if we have at least 2 submissions)
        median_score = features_df['score'].median()
        high_scoring = [s for s in submissions if s['score'] >= median_score]
        low_scoring = [s for s in submissions if s['score'] < median_score]
        
        plt.figure(figsize=(15, 10))
        
        # Distribution plots
        plt.subplot(2, 2, 1)
        for sub in high_scoring[:5]:  # Limit to 5 for readability
            sns.kdeplot(sub['df']['rainfall'], alpha=0.5, label=f"Score: {sub['score']:.4f}")
        plt.title('High-Scoring Submissions Distribution')
        plt.xlabel('Rainfall')
        plt.ylabel('Density')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        for sub in low_scoring[:5]:  # Limit to 5 for readability
            sns.kdeplot(sub['df']['rainfall'], alpha=0.5, label=f"Score: {sub['score']:.4f}")
        plt.title('Low-Scoring Submissions Distribution')
        plt.xlabel('Rainfall')
        plt.ylabel('Density')
        plt.legend()
        
        # Box plots
        plt.subplot(2, 2, 3)
        boxplot_data = []
        labels = []
        for sub in submissions[:10]:  # Limit to 10 submissions
            boxplot_data.append(sub['df']['rainfall'])
            labels.append(f"{sub['score']:.4f}")
        
        plt.boxplot(boxplot_data, labels=labels)
        plt.title('Boxplot by Score (Highest to Lowest)')
        plt.xlabel('Score')
        plt.ylabel('Rainfall')
        plt.xticks(rotation=45)
        
        # ECDF comparison
        plt.subplot(2, 2, 4)
        for sub in submissions[:5]:  # Limit to 5 for clarity
            sorted_data = np.sort(sub['df']['rainfall'])
            ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            plt.plot(sorted_data, ecdf, label=f"Score: {sub['score']:.4f}")
        
        plt.title('ECDF by Score')
        plt.xlabel('Rainfall')
        plt.ylabel('ECDF')
        plt.legend(title='Score')
        
        plt.tight_layout()
        plt.savefig('analysis/scores/high_vs_low_comparison.png')
    
    # Generate key insights
    # 1. Find the features that most strongly differentiate high from low scores
    top_differentiators = score_correlations.abs().sort_values(ascending=False).head(5).index
    
    # 2. Calculate the optimal range for each top feature
    optimal_ranges = {}
    for feature in top_differentiators:
        # Use top 3 submissions to determine optimal range
        top_values = [features_df[features_df['score'] == s['score']][feature].values[0] for s in submissions[:3]]
        optimal_ranges[feature] = {
            'mean': np.mean(top_values),
            'min': min(top_values),
            'max': max(top_values)
        }
    
    # Generate a report file
    report_path = 'analysis/scores/score_insights.txt'
    with open(report_path, 'w') as f:
        f.write("=== SCORE ANALYSIS INSIGHTS ===\n\n")
        
        f.write("Top features correlated with higher scores:\n")
        for feature, corr in score_correlations.head(10).items():
            f.write(f"- {feature}: {corr:+.4f}\n")
        
        f.write("\nFeatures correlated with lower scores:\n")
        for feature, corr in score_correlations.tail(10).items():
            f.write(f"- {feature}: {corr:+.4f}\n")
        
        f.write("\nOptimal ranges for top features (based on highest-scoring submissions):\n")
        for feature, ranges in optimal_ranges.items():
            f.write(f"- {feature}: {ranges['min']:.4f} to {ranges['max']:.4f} (mean: {ranges['mean']:.4f})\n")
        
        f.write("\nRecommendations for improving scores:\n")
        
        # Generate specific recommendations based on correlations
        if 'std' in top_differentiators:
            if score_correlations['std'] > 0:
                f.write("- Increase variability in predictions (higher standard deviation)\n")
            else:
                f.write("- Decrease variability in predictions (lower standard deviation)\n")
                
        if 'mean' in top_differentiators:
            target_mean = optimal_ranges.get('mean', {}).get('mean', 0.5)
            f.write(f"- Aim for a mean prediction value around {target_mean:.4f}\n")
            
        if 'unique_ratio' in top_differentiators:
            if score_correlations['unique_ratio'] > 0:
                f.write("- Increase the number of unique prediction values\n")
            else:
                f.write("- Use fewer unique prediction values\n")
                
        if 'skew' in top_differentiators:
            target_skew = optimal_ranges.get('skew', {}).get('mean', 0)
            if target_skew > 0.1:
                f.write("- Give predictions a positive skew (more mass on lower values)\n")
            elif target_skew < -0.1:
                f.write("- Give predictions a negative skew (more mass on higher values)\n")
            else:
                f.write("- Aim for a more symmetric distribution\n")
        
        if 'out_of_range' in top_differentiators:
            if score_correlations['out_of_range'] > 0:
                f.write("- Consider allowing some values outside 0-1 range\n")
            else:
                f.write("- Ensure all values are within 0-1 range\n")
    
    print(f"\nAnalysis complete! Insights saved to {report_path}")
    print("Key visualizations saved to analysis/scores/ directory")
    
    # Return the insights dictionary for further use
    return {
        'correlations': score_correlations,
        'optimal_ranges': optimal_ranges,
        'features_df': features_df
    }

def apply_recommendations(best_submission_path, insights):
    """Apply recommended transformations to improve a submission based on insights"""
    print(f"\n=== APPLYING RECOMMENDATIONS TO {best_submission_path} ===")
    
    if not os.path.exists(best_submission_path):
        print(f"Error: {best_submission_path} not found")
        return None
    
    # Load the best submission
    best_df = pd.read_csv(best_submission_path)
    
    # Get the optimal ranges from insights
    optimal_ranges = insights['optimal_ranges']
    correlations = insights['correlations']
    
    # Calculate current statistics
    rainfall = best_df['rainfall'].values
    current_stats = {
        'mean': np.mean(rainfall),
        'std': np.std(rainfall),
        'min': np.min(rainfall),
        'max': np.max(rainfall),
        'skew': stats.skew(rainfall),
        'kurtosis': stats.kurtosis(rainfall)
    }
    
    print("\nCurrent statistics:")
    for stat, value in current_stats.items():
        print(f"{stat}: {value:.4f}")
    
    print("\nTarget ranges (from top submissions):")
    for feature, ranges in optimal_ranges.items():
        print(f"{feature}: {ranges['min']:.4f} to {ranges['max']:.4f} (mean: {ranges['mean']:.4f})")
    
    # Create a copy for transformation
    improved_df = best_df.copy()
    
    # Apply transformations based on recommendations
    transformed_rainfall = rainfall.copy()
    
    # 1. Adjust mean if needed
    if 'mean' in optimal_ranges:
        target_mean = optimal_ranges['mean']['mean']
        current_mean = current_stats['mean']
        
        if abs(current_mean - target_mean) > 0.05:
            print(f"\nAdjusting mean from {current_mean:.4f} to {target_mean:.4f}")
            shift = target_mean - current_mean
            transformed_rainfall = transformed_rainfall + shift
    
    # 2. Adjust standard deviation if needed
    if 'std' in optimal_ranges:
        target_std = optimal_ranges['std']['mean']
        current_std = current_stats['std']
        
        if abs(current_std - target_std) > 0.05:
            print(f"Adjusting standard deviation from {current_std:.4f} to {target_std:.4f}")
            # Scale around the mean
            mean = np.mean(transformed_rainfall)
            transformed_rainfall = (transformed_rainfall - mean) * (target_std / current_std) + mean
    
    # 3. Adjust skew if needed
    if 'skew' in optimal_ranges:
        target_skew = optimal_ranges['skew']['mean']
        current_skew = stats.skew(transformed_rainfall)
        
        if abs(current_skew - target_skew) > 0.1:
            print(f"Adjusting skew from {current_skew:.4f} towards {target_skew:.4f}")
            
            if current_skew < target_skew:
                # Increase skew by applying a power < 1
                power = 0.5
                transformed_rainfall = np.power(transformed_rainfall, power)
            else:
                # Decrease skew by applying a power > 1
                power = 2.0
                transformed_rainfall = np.power(transformed_rainfall, power)
            
            # Rescale to [0, 1] range
            min_val = np.min(transformed_rainfall)
            max_val = np.max(transformed_rainfall)
            transformed_rainfall = (transformed_rainfall - min_val) / (max_val - min_val)
    
    # 4. Ensure values are within range
    if correlations['out_of_range'] < 0:
        transformed_rainfall = np.clip(transformed_rainfall, 0, 1)
        print("Clipped values to 0-1 range")
    
    # Apply transformed values
    improved_df['rainfall'] = transformed_rainfall
    
    # Calculate new statistics
    new_stats = {
        'mean': np.mean(improved_df['rainfall']),
        'std': np.std(improved_df['rainfall']),
        'min': np.min(improved_df['rainfall']),
        'max': np.max(improved_df['rainfall']),
        'skew': stats.skew(improved_df['rainfall']),
        'kurtosis': stats.kurtosis(improved_df['rainfall'])
    }
    
    print("\nNew statistics after transformation:")
    for stat, value in new_stats.items():
        print(f"{stat}: {value:.4f}")
    
    # Save improved submission
    output_path = best_submission_path.replace('.csv', '_improved.csv')
    improved_df.to_csv(output_path, index=False)
    print(f"\nSaved improved submission to {output_path}")
    
    # Create visualization comparing before and after
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    sns.kdeplot(best_df['rainfall'], label='Original')
    sns.kdeplot(improved_df['rainfall'], label='Improved')
    plt.title('Distribution Comparison')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.scatter(best_df['rainfall'], improved_df['rainfall'], alpha=0.2)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Original Values')
    plt.ylabel('Improved Values')
    plt.title('Original vs Improved Values')
    
    plt.tight_layout()
    plt.savefig('analysis/scores/improved_submission_comparison.png')
    print("Saved visualization to analysis/scores/improved_submission_comparison.png")
    
    return improved_df

if __name__ == "__main__":
    print("=== SCORE PATTERN ANALYSIS AND IMPROVEMENT ===")
    insights = analyze_submission_scores()
    
    if insights:
        print("\n=== KEY TAKEAWAYS ===")
        top_pos = insights['correlations'].head(3).to_dict()
        top_neg = insights['correlations'].tail(3).to_dict()
        
        print("Features most positively correlated with higher scores:")
        for feature, corr in top_pos.items():
            print(f"- {feature}: {corr:+.4f}")
            
        print("\nFeatures most negatively correlated with higher scores:")
        for feature, corr in top_neg.items():
            print(f"- {feature}: {corr:+.4f}")
        
        # If we have a best submission file, apply improvements
        best_submission_path = '/home/USER/Projects/RAIN/data/keep/submission_86403.csv'
        if os.path.exists(best_submission_path):
            print("\nFound best submission. Applying recommendations...")
            improved_df = apply_recommendations(best_submission_path, insights)
            
            # Copy to final directory
            os.makedirs('data/final', exist_ok=True)
            final_path = 'data/final/submission_final_optimized.csv'
            improved_df.to_csv(final_path, index=False)
            print(f"Copied optimized submission to {final_path}")
        else:
            print("\nBest submission file not found. Please specify the path to improve a submission.")
