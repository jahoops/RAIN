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
            'filename': sub['filename'],
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
        
        features.append(stats_dict)
    
    # Convert to DataFrame for easier analysis
    features_df = pd.DataFrame(features)
    
    # Calculate correlations with score
    score_correlations = features_df.corr()['score'].drop('score').sort_values(ascending=False)
    
    print("\n=== CORRELATIONS WITH SCORE ===")
    for feature, corr in score_correlations.items():
        print(f"{feature:<15}: {corr:+.4f}")
    
    # Create visualizations
    
    # 1. Correlation heatmap
    plt.figure(figsize=(14, 12))
    corr_matrix = features_df.drop('filename', axis=1).corr()
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
        for sub in high_scoring:
            sns.kdeplot(sub['df']['rainfall'], alpha=0.5, label=f"{sub['filename']} ({sub['score']:.4f})")
        plt.title('High-Scoring Submissions Distribution')
        plt.xlabel('Rainfall')
        plt.ylabel('Density')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        for sub in low_scoring:
            sns.kdeplot(sub['df']['rainfall'], alpha=0.5, label=f"{sub['filename']} ({sub['score']:.4f})")
        plt.title('Low-Scoring Submissions Distribution')
        plt.xlabel('Rainfall')
        plt.ylabel('Density')
        plt.legend()
        
        # Box plots
        plt.subplot(2, 2, 3)
        boxplot_data = []
        labels = []
        for sub in submissions:
            boxplot_data.append(sub['df']['rainfall'])
            labels.append(f"{sub['score']:.4f}")
        
        plt.boxplot(boxplot_data, labels=labels)
        plt.title('Boxplot by Score (Highest to Lowest)')
        plt.xlabel('Score')
        plt.ylabel('Rainfall')
        plt.xticks(rotation=45)
        
        # ECDF comparison
        plt.subplot(2, 2, 4)
        for sub in submissions:
            sorted_data = np.sort(sub['df']['rainfall'])
            ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            plt.plot(sorted_data, ecdf, label=f"{sub['score']:.4f}")
        
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
    
    # Create a recommended submission template based on insights
    best_submission = submissions[0]['df'].copy()
    
    # Return the insights dictionary for further use
    return {
        'correlations': score_correlations,
        'optimal_ranges': optimal_ranges,
        'features_df': features_df
    }

if __name__ == "__main__":
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
