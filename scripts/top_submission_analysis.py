import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_submission(file_path):
    """Extract statistical signature from a submission file"""
    df = pd.read_csv(file_path)
    rainfall = df['rainfall']
    
    # Calculate key statistics
    stats = {
        'mean': rainfall.mean(),
        'median': rainfall.median(),
        'std': rainfall.std(),
        'min': rainfall.min(),
        'max': rainfall.max(),
        'range': rainfall.max() - rainfall.min(),
        'zero_count': (rainfall == 0).sum(),
        'zero_pct': (rainfall == 0).mean() * 100,
        'non_zero_mean': rainfall[rainfall > 0].mean() if any(rainfall > 0) else 0,
        'unique_values': len(rainfall.unique()),
        'unique_ratio': len(rainfall.unique()) / len(rainfall) * 100,
        'below_0.1': (rainfall < 0.1).sum(),
        'below_0.1_pct': (rainfall < 0.1).mean() * 100,
        '0.1_to_0.5': ((rainfall >= 0.1) & (rainfall < 0.5)).sum(),
        '0.5_to_0.9': ((rainfall >= 0.5) & (rainfall < 0.9)).sum(),
        'above_0.9': (rainfall >= 0.9).sum(),
        'above_0.9_pct': (rainfall >= 0.9).mean() * 100,
        'exactly_1.0': (rainfall == 1.0).sum(),
    }
    
    # Distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(rainfall, bins=50, alpha=0.7)
    plt.title(f"Distribution for {os.path.basename(file_path)} (Score: Check submission_results.csv)")
    plt.xlabel('Rainfall')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    save_path = f"analysis_{os.path.basename(file_path).replace('.csv', '.png')}"
    plt.savefig(save_path)
    plt.close()
    
    return stats

# List of submissions to compare
submissions = [
    '/home/USER/Projects/RAIN/data/Actual/submission.86403.csv',  # Top scoring
    '/home/USER/Projects/RAIN/data/candidates/continuous_winner_1741191165.csv',  # Our latest
]

# Compare submissions
results = {}
for submission in submissions:
    if os.path.exists(submission):
        stats = analyze_submission(submission)
        results[os.path.basename(submission)] = stats
    else:
        print(f"File not found: {submission}")

# Print comparison
print("\nSubmission Comparison:")
if results:
    # Get column width
    key_width = max(len(key) for key in list(results.values())[0].keys())
    file_width = max(len(file) for file in results.keys())
    
    # Print headers
    print(f"{'Statistic':{key_width}} | " + " | ".join(f"{file:{file_width}}" for file in results.keys()))
    print("-" * (key_width + sum(file_width + 3 for _ in results)))
    
    # Print each statistic
    for key in list(results.values())[0].keys():
        print(f"{key:{key_width}} | " + " | ".join(f"{results[file][key]:{file_width}.6f}" if isinstance(results[file][key], float) else f"{results[file][key]:{file_width}}" for file in results.keys()))