import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def load_synthetic_data() -> pd.DataFrame:
    """Load the improved synthetic test data"""
    synthetic_path = os.path.join('data', 'improved_synthetic_test.csv')
    if not os.path.exists(synthetic_path):
        raise FileNotFoundError(
            f"Synthetic test data not found at {synthetic_path}. "
            "Please run synthetic_validation_report.py first."
        )
    return pd.read_csv(synthetic_path)

def load_benchmark() -> pd.DataFrame:
    """Load the benchmark submission for comparison"""
    # First look in keep directory
    benchmark_path = os.path.join('data', 'keep', 'submission_all_score_0.882270.csv')
    if not os.path.exists(benchmark_path):
        # Try main data directory as fallback
        benchmark_path = os.path.join('data', 'submission_all_score_0.882270.csv')
        if not os.path.exists(benchmark_path):
            raise FileNotFoundError(
                "Benchmark submission not found. Please ensure submission_all_score_0.882270.csv "
                "exists in either data/keep/ or data/ directory."
            )
    return pd.read_csv(benchmark_path)

def evaluate_against_synthetic(submission_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict:
    """Evaluate a submission against synthetic data"""
    # Find common IDs between submission and synthetic data
    common_ids = set(submission_df['id']).intersection(set(synthetic_df['id']))
    
    results = {
        'common_ids': len(common_ids),
        'correlation': None,
        'auc': None,
        'mse': None,
        'match_quality': None
    }
    
    if common_ids:
        # Filter to common IDs and ensure same order
        sub_common = submission_df[submission_df['id'].isin(common_ids)].copy().sort_values('id')
        syn_common = synthetic_df[synthetic_df['id'].isin(common_ids)].copy().sort_values('id')
        
        # Calculate correlation
        results['correlation'] = np.corrcoef(syn_common['rainfall'], sub_common['rainfall'])[0, 1]
        
        # Calculate AUC
        if syn_common['rainfall'].nunique() > 2:
            binary_target = (syn_common['rainfall'] > syn_common['rainfall'].median()).astype(int)
        else:
            binary_target = syn_common['rainfall']
        try:
            results['auc'] = roc_auc_score(binary_target, sub_common['rainfall'])
        except:
            results['auc'] = None
        
        # Calculate MSE
        results['mse'] = np.mean((syn_common['rainfall'] - sub_common['rainfall'])**2)
        
        # Calculate match quality
        results['match_quality'] = len(common_ids) / len(synthetic_df) * max(0, results['correlation'])
    
    return results

def estimate_actual_score(synthetic_metrics: Dict, benchmark_metrics: Dict) -> float:
    """Estimate the actual contest score based on synthetic metrics comparison with benchmark"""
    # Benchmark's known actual score
    benchmark_actual = 0.85518
    
    # Calculate relative performance compared to benchmark (using correlation as primary metric)
    if benchmark_metrics['correlation'] > 0:
        relative_performance = synthetic_metrics['correlation'] / benchmark_metrics['correlation']
        
        # Apply squared relationship to account for the non-linear relationship between 
        # synthetic metrics and actual performance
        estimated_score = benchmark_actual * (relative_performance ** 0.9)
        
        # Clip to realistic range
        estimated_score = max(0.5, min(0.9, estimated_score))
    else:
        # Fallback if benchmark correlation is not positive
        estimated_score = 0.5
    
    return estimated_score

def plot_comparison(submission_df: pd.DataFrame, synthetic_df: pd.DataFrame, 
                   benchmark_df: pd.DataFrame, results: Dict, benchmark_results: Dict) -> None:
    """Create visualizations comparing the submission against synthetic data and benchmark"""
    # Create plots directory if needed
    os.makedirs('plots', exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # 1. Plot rainfall distribution comparison
    plt.subplot(2, 2, 1)
    common_ids = set(submission_df['id']).intersection(set(synthetic_df['id']))
    
    if common_ids:
        sub_common = submission_df[submission_df['id'].isin(common_ids)].sort_values('id')
        syn_common = synthetic_df[synthetic_df['id'].isin(common_ids)].sort_values('id')
        bench_common = benchmark_df[benchmark_df['id'].isin(common_ids)].sort_values('id')
        
        plt.hist(syn_common['rainfall'], bins=20, alpha=0.5, label='Synthetic Truth')
        plt.hist(sub_common['rainfall'], bins=20, alpha=0.5, label='Your Submission')
        plt.hist(bench_common['rainfall'], bins=20, alpha=0.5, label='Benchmark')
        plt.title('Rainfall Distribution Comparison')
        plt.xlabel('Rainfall Value')
        plt.ylabel('Count')
        plt.legend()
    
    # 2. Plot scatter plot between synthetic and submission
    plt.subplot(2, 2, 2)
    if common_ids:
        plt.scatter(syn_common['rainfall'], sub_common['rainfall'], alpha=0.5, 
                   label=f'Your Sub (r={results["correlation"]:.4f})')
        plt.scatter(syn_common['rainfall'], bench_common['rainfall'], alpha=0.5, 
                   label=f'Benchmark (r={benchmark_results["correlation"]:.4f})')
        plt.xlabel('Synthetic Truth')
        plt.ylabel('Prediction')
        plt.title('Predictions vs Synthetic Truth')
        plt.legend()
        
        # Add correlation line for visual clarity
        z = np.polyfit(syn_common['rainfall'], sub_common['rainfall'], 1)
        p = np.poly1d(z)
        plt.plot(np.sort(syn_common['rainfall']), p(np.sort(syn_common['rainfall'])), 
                "r--", alpha=0.5)
    
    # 3. Plot metric comparison
    plt.subplot(2, 2, 3)
    metrics = ['correlation', 'auc', 'match_quality']
    x = np.arange(len(metrics))
    width = 0.35
    
    your_values = [results[m] for m in metrics]
    benchmark_values = [benchmark_results[m] for m in metrics]
    
    plt.bar(x - width/2, your_values, width, label='Your Submission')
    plt.bar(x + width/2, benchmark_values, width, label='Benchmark')
    plt.xticks(x, metrics)
    plt.title('Metric Comparison')
    plt.legend()
    
    # 4. Add summary text
    plt.subplot(2, 2, 4)
    plt.axis('off')
    estimated_score = estimate_actual_score(results, benchmark_results)
    
    summary_text = f"""
    SUBMISSION VALIDATION SUMMARY
    
    Synthetic Metrics:
    - Correlation: {results['correlation']:.4f} (Benchmark: {benchmark_results['correlation']:.4f})
    - AUC: {results['auc']:.4f} (Benchmark: {benchmark_results['auc']:.4f})
    - Match Quality: {results['match_quality']:.4f} (Benchmark: {benchmark_results['match_quality']:.4f})
    - Common IDs: {results['common_ids']} (Benchmark: {benchmark_results['common_ids']})
    
    Performance vs Benchmark:
    - Correlation: {100 * results['correlation'] / benchmark_results['correlation']:.1f}% of benchmark
    - AUC: {100 * results['auc'] / benchmark_results['auc']:.1f}% of benchmark
    
    Estimated Contest Score: {estimated_score:.4f}
    (Benchmark actual score: 0.85518)
    """
    plt.text(0, 0.5, summary_text, fontsize=12, verticalalignment='center')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('plots/submission_validation.png')
    print(f"Saved comparison plot to plots/submission_validation.png")
    
    # Return without showing the plot in non-interactive mode
    plt.close()

def validate_submission(submission_path: str, quiet: bool = False) -> Dict:
    """
    Validate a submission file against synthetic data and benchmark
    
    Args:
        submission_path: Path to the submission CSV file
        quiet: If True, suppress detailed output
        
    Returns:
        Dictionary with validation results
    """
    if not quiet:
        print(f"Validating submission: {submission_path}")
    
    # Load data files
    try:
        submission_df = pd.read_csv(submission_path)
        synthetic_df = load_synthetic_data()
        benchmark_df = load_benchmark()
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return {}
    
    # Check submission format
    if 'id' not in submission_df.columns or 'rainfall' not in submission_df.columns:
        print("Error: Submission must contain 'id' and 'rainfall' columns")
        return {}
    
    # Evaluate against synthetic data
    results = evaluate_against_synthetic(submission_df, synthetic_df)
    benchmark_results = evaluate_against_synthetic(benchmark_df, synthetic_df)
    
    # Calculate estimated actual score
    estimated_score = estimate_actual_score(results, benchmark_results)
    results['estimated_score'] = estimated_score
    
    if not quiet:
        # Create validation plot
        plot_comparison(submission_df, synthetic_df, benchmark_df, results, benchmark_results)
        
        # Print validation results
        print("\n=== VALIDATION RESULTS ===")
        print(f"Common IDs with synthetic data: {results['common_ids']}")
        print(f"Correlation with synthetic: {results['correlation']:.6f} (Benchmark: {benchmark_results['correlation']:.6f})")
        if results['auc']:
            print(f"AUC: {results['auc']:.6f} (Benchmark: {benchmark_results['auc']:.6f})")
        print(f"MSE: {results['mse']:.6f} (Benchmark: {benchmark_results['mse']:.6f})")
        print(f"Match quality: {results['match_quality']:.6f} (Benchmark: {benchmark_results['match_quality']:.6f})")
        print("\n=== PERFORMANCE ESTIMATE ===")
        print(f"Estimated actual contest score: {estimated_score:.6f}")
        
        # Add performance tier
        if estimated_score >= 0.85:
            tier = "EXCELLENT - On par with benchmark!"
        elif estimated_score >= 0.82:
            tier = "VERY GOOD - Close to benchmark performance"
        elif estimated_score >= 0.80:
            tier = "GOOD - Competitive but room for improvement"
        elif estimated_score >= 0.75:
            tier = "AVERAGE - More optimization needed"
        else:
            tier = "BELOW AVERAGE - Significant improvement needed"
            
        print(f"Performance tier: {tier}")
        
        # Add helpful suggestions based on results
        print("\n=== RECOMMENDATIONS ===")
        if results['correlation'] < benchmark_results['correlation'] * 0.95:
            print("- Your model's predictions differ from the benchmark. Consider analyzing feature importance.")
        
        if results['common_ids'] < synthetic_df.shape[0] * 0.7:
            print("- Low ID overlap with synthetic data could affect validation accuracy.")
        
        print("- Compare your model's feature importance with benchmark model if available.")
        print("- Consider ensemble methods combining your model with other approaches.")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Validate submissions against synthetic data")
    parser.add_argument("submission", help="Path to submission CSV file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress detailed output")
    args = parser.parse_args()
    
    validate_submission(args.submission, args.quiet)

if __name__ == "__main__":
    main()
