import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import glob
import os

# Configuration
RESULTS_DIR = "feature_evaluation_results"
ANALYSIS_DIR = "error_analysis"
os.makedirs(ANALYSIS_DIR, exist_ok=True)
os.makedirs(f"{ANALYSIS_DIR}/plots", exist_ok=True)

def analyze_prediction_errors(feature_set_name=None):
    """
    Analyze prediction errors for a specific feature set or all feature sets.
    If feature_set_name is None, analyzes all available feature sets.
    """
    # Load training data for feature analysis
    train = pd.read_csv('/home/USER/Projects/RAIN/data/train.csv')
    
    if feature_set_name:
        prediction_files = [f"{RESULTS_DIR}/{feature_set_name}_predictions.csv"]
    else:
        # Find all prediction files
        prediction_files = glob.glob(f"{RESULTS_DIR}/*_predictions.csv")
    
    for pred_file in prediction_files:
        # Extract feature set name from filename
        name = os.path.basename(pred_file).replace('_predictions.csv', '')
        print(f"\n{'-'*50}")
        print(f"Analyzing prediction errors for: {name}")
        print(f"{'-'*50}")
        
        # Load predictions
        predictions = pd.read_csv(pred_file)
        
        # Merge with original training data
        analysis_df = train.copy()
        analysis_df['predicted'] = predictions['predicted']
        analysis_df['error'] = predictions['error']
        analysis_df['abs_error'] = predictions['abs_error']
        
        # 1. Error distribution statistics
        error_stats = {
            'mean_error': analysis_df['error'].mean(),
            'median_error': analysis_df['error'].median(),
            'std_error': analysis_df['error'].std(),
            'mean_abs_error': analysis_df['abs_error'].mean(),
            'median_abs_error': analysis_df['abs_error'].median(),
            'max_abs_error': analysis_df['abs_error'].max(),
            'min_abs_error': analysis_df['abs_error'].min(),
        }
        
        # Save error statistics
        pd.DataFrame([error_stats]).to_csv(f"{ANALYSIS_DIR}/{name}_error_stats.csv", index=False)
        
        # 2. Analyze what correlates with prediction errors
        feature_cols = [col for col in analysis_df.columns if col not in 
                        ['id', 'rainfall', 'predicted', 'error', 'abs_error']]
        
        # Calculate correlation between features and absolute error
        correlations = []
        for col in feature_cols:
            if analysis_df[col].nunique() > 1:  # Skip constant columns
                corr, _ = spearmanr(analysis_df[col], analysis_df['abs_error'])
                correlations.append({
                    'feature': col,
                    'correlation_with_error': corr
                })
        
        # Create correlation DataFrame
        corr_df = pd.DataFrame(correlations).sort_values('correlation_with_error', 
                                                         key=abs, ascending=False)
        corr_df.to_csv(f"{ANALYSIS_DIR}/{name}_error_correlations.csv", index=False)
        
        # 3. Identify where the model performs well vs poorly
        analysis_df['error_quantile'] = pd.qcut(analysis_df['abs_error'], 5, labels=False)
        performance_groups = {
            'best': analysis_df[analysis_df['error_quantile'] == 0],
            'worst': analysis_df[analysis_df['error_quantile'] == 4]
        }
        
        # Compare best vs worst predictions
        comparison = {}
        for col in feature_cols:
            if analysis_df[col].nunique() > 1:
                comparison[col] = {
                    'best_mean': performance_groups['best'][col].mean(),
                    'worst_mean': performance_groups['worst'][col].mean(),
                    'difference': performance_groups['best'][col].mean() - performance_groups['worst'][col].mean(),
                    'best_std': performance_groups['best'][col].std(),
                    'worst_std': performance_groups['worst'][col].std()
                }
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison).T
        comparison_df['abs_difference'] = comparison_df['difference'].abs()
        comparison_df = comparison_df.sort_values('abs_difference', ascending=False)
        comparison_df.to_csv(f"{ANALYSIS_DIR}/{name}_performance_comparison.csv", index=True)
        
        # 4. Visualizations
        
        # 4.1. Error distribution by actual value
        plt.figure(figsize=(10, 6))
        plt.scatter(analysis_df['rainfall'], analysis_df['abs_error'], alpha=0.5)
        plt.xlabel('Actual Rainfall')
        plt.ylabel('Absolute Error')
        plt.title(f'{name}: Error Distribution by Actual Rainfall')
        plt.savefig(f"{ANALYSIS_DIR}/plots/{name}_error_by_actual.png")
        plt.close()
        
        # 4.2. Top features correlated with error
        top_corr_features = corr_df.head(10)
        plt.figure(figsize=(10, 6))
        plt.barh(top_corr_features['feature'][::-1], 
                top_corr_features['correlation_with_error'][::-1])
        plt.xlabel('Correlation with Absolute Error')
        plt.title(f'{name}: Top Features Correlated with Prediction Error')
        plt.savefig(f"{ANALYSIS_DIR}/plots/{name}_error_correlations.png")
        plt.close()
        
        # 4.3. Feature value distribution in best vs worst predictions
        for col in comparison_df.head(5).index:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(performance_groups['best'][col], label='Best Predictions')
            sns.kdeplot(performance_groups['worst'][col], label='Worst Predictions')
            plt.xlabel(col)
            plt.ylabel('Density')
            plt.title(f'{name}: {col} Distribution in Best vs. Worst Predictions')
            plt.legend()
            plt.savefig(f"{ANALYSIS_DIR}/plots/{name}_{col}_best_worst_dist.png")
            plt.close()
        
        # 5. Analyze error patterns by prediction range
        analysis_df['prediction_bin'] = pd.cut(analysis_df['predicted'], 10)
        bin_stats = analysis_df.groupby('prediction_bin').agg({
            'error': ['mean', 'std'],
            'abs_error': ['mean', 'median', 'count']
        })
        bin_stats.to_csv(f"{ANALYSIS_DIR}/{name}_error_by_prediction_range.csv")
        
        # Visualize error patterns by prediction range
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(bin_stats)), bin_stats[('abs_error', 'mean')], 
                yerr=bin_stats[('error', 'std')])
        plt.xticks(range(len(bin_stats)), 
                  [str(x) for x in bin_stats.index], rotation=45)
        plt.xlabel('Predicted Value Range')
        plt.ylabel('Mean Absolute Error')
        plt.title(f'{name}: Error by Prediction Range')
        plt.tight_layout()
        plt.savefig(f"{ANALYSIS_DIR}/plots/{name}_error_by_prediction_bin.png")
        plt.close()
        
        print(f"Analysis completed for {name}")
        print(f"Error statistics: Mean Abs Error = {error_stats['mean_abs_error']:.4f}")
        print(f"Top features correlated with error:")
        print(corr_df.head(5).to_string(index=False))
        print(f"Results saved to {ANALYSIS_DIR}/{name}_*.csv and {ANALYSIS_DIR}/plots/")

if __name__ == "__main__":
    # Analyze all available prediction files
    analyze_prediction_errors()