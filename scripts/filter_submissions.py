import pandas as pd
import numpy as np
import os
import glob
import shutil
from sklearn.metrics import roc_auc_score

def filter_submissions():
    """Filter out low-quality submissions that don't correlate well with the best model"""
    print("=== FILTERING SUBMISSIONS ===")
    
    # Reference submission
    reference_path = 'data/keep/submission_all_score_0.882270.csv'
    if not os.path.exists(reference_path):
        print(f"Reference submission not found at {reference_path}")
        reference_paths = glob.glob('data/keep/submission_*_score_*.csv')
        if reference_paths:
            # Sort by score in filename
            reference_paths.sort(key=lambda x: float(x.split('score_')[1].split('_')[0].split('.csv')[0]), reverse=True)
            reference_path = reference_paths[0]
            print(f"Using best available reference: {reference_path}")
        else:
            print("No suitable reference submission found. Cannot continue.")
            return
    
    # Load reference submission
    reference_df = pd.read_csv(reference_path)
    print(f"Loaded reference submission with {len(reference_df)} rows")
    
    # Find all submissions in keep directory
    keep_dir = 'data/keep'
    submission_files = glob.glob(os.path.join(keep_dir, 'submission_*.csv'))
    print(f"Found {len(submission_files)} submission files in {keep_dir}")
    
    # Create directory for low-quality submissions
    low_quality_dir = os.path.join(keep_dir, 'low_quality')
    os.makedirs(low_quality_dir, exist_ok=True)
    
    # Analyze each submission and compare with reference
    results = []
    for file_path in submission_files:
        filename = os.path.basename(file_path)
        
        # Skip the reference itself
        if file_path == reference_path:
            print(f"Skipping reference submission: {filename}")
            continue
        
        try:
            df = pd.read_csv(file_path)
            
            # Find common IDs
            common_ids = set(df['id']).intersection(set(reference_df['id']))
            if len(common_ids) < 50:
                print(f"WARNING: {filename} has only {len(common_ids)} IDs in common with reference")
                correlation = 0
                auc = 0.5  # Default for very few samples
            else:
                # Filter to common IDs and sort
                df_common = df[df['id'].isin(common_ids)].sort_values('id')
                ref_common = reference_df[reference_df['id'].isin(common_ids)].sort_values('id')
                
                # Calculate correlation
                correlation = np.corrcoef(df_common['rainfall'], ref_common['rainfall'])[0, 1]
                
                # Calculate agreement score (AUC using one to predict the other)
                df_binary = (df_common['rainfall'] > 0.5).astype(int)
                ref_binary = (ref_common['rainfall'] > 0.5).astype(int)
                try:
                    auc = roc_auc_score(ref_binary, df_common['rainfall'])
                except:
                    auc = 0.5
            
            # Calculate mean value and standard deviation
            mean_val = df['rainfall'].mean()
            std_val = df['rainfall'].std()
            
            # Store results
            quality_score = (correlation + auc - 0.5) / 2  # Combined score between 0 and 1
            
            results.append({
                'filename': filename,
                'path': file_path,
                'correlation': correlation,
                'auc': auc,
                'quality_score': quality_score,
                'mean': mean_val,
                'std_dev': std_val,
                'common_ids': len(common_ids)
            })
            
            # Print quick assessment
            quality_level = "GOOD" if quality_score > 0.7 else "MEDIUM" if quality_score > 0.5 else "POOR"
            print(f"{filename}: Quality={quality_score:.4f}, Corr={correlation:.4f}, AUC={auc:.4f} - {quality_level}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # Sort results by quality score
    results.sort(key=lambda x: x['quality_score'], reverse=True)
    
    # Print summary
    print("\n=== SUBMISSION QUALITY SUMMARY ===")
    print(f"Total submissions analyzed: {len(results)}")
    
    high_quality = [r for r in results if r['quality_score'] > 0.7]
    medium_quality = [r for r in results if 0.5 < r['quality_score'] <= 0.7]
    low_quality = [r for r in results if r['quality_score'] <= 0.5]
    
    print(f"High quality submissions: {len(high_quality)}")
    print(f"Medium quality submissions: {len(medium_quality)}")
    print(f"Low quality submissions: {len(low_quality)}")
    
    # Create CSV report
    report_df = pd.DataFrame(results)
    report_df.to_csv('analysis/submission_quality.csv', index=False)
    print("\nDetailed report saved to analysis/submission_quality.csv")
    
    # Handle low quality submissions
    print("\n=== LOW QUALITY SUBMISSIONS ===")
    for submission in low_quality:
        print(f"{submission['filename']}: Score={submission['quality_score']:.4f}, Corr={submission['correlation']:.4f}")
        
        # Move to low_quality directory
        source_path = submission['path']
        dest_path = os.path.join(low_quality_dir, submission['filename'])
        shutil.move(source_path, dest_path)
        print(f"Moved to {low_quality_dir}/")
    
    # Generate optimized ensemble recommendations
    print("\n=== OPTIMIZED ENSEMBLE RECOMMENDATIONS ===")
    
    # Use only high quality submissions
    if high_quality:
        top_submissions = [r['filename'] for r in high_quality[:5]]
        print("Best 5 submissions for ensemble:")
        for sub in top_submissions:
            print(f"  - {sub}")
            
        # Find submissions with lowest correlation among high quality ones
        if len(high_quality) > 1:
            # Calculate average correlation between each submission
            unique_submissions = []
            for i, s1 in enumerate(high_quality[:10]):  # Limit to top 10
                corrs = []
                for j, s2 in enumerate(high_quality[:10]):
                    if i != j:
                        # Load submissions
                        df1 = pd.read_csv(os.path.join(keep_dir, s1['filename']))
                        df2 = pd.read_csv(os.path.join(keep_dir, s2['filename']))
                        
                        # Find common IDs
                        common_ids = set(df1['id']).intersection(set(df2['id']))
                        if len(common_ids) > 50:
                            df1_common = df1[df1['id'].isin(common_ids)].sort_values('id')
                            df2_common = df2[df2['id'].isin(common_ids)].sort_values('id')
                            corr = np.corrcoef(df1_common['rainfall'], df2_common['rainfall'])[0, 1]
                            corrs.append(corr)
                
                if corrs:
                    avg_corr = np.mean(corrs)
                    unique_submissions.append((s1['filename'], avg_corr))
            
            # Sort by lowest correlation (most unique)
            unique_submissions.sort(key=lambda x: x[1])
            
            print("\nMost unique (diverse) high-quality submissions:")
            for sub, corr in unique_submissions[:3]:
                print(f"  - {sub} (avg correlation: {corr:.4f})")
            
            print("\nRecommended ensemble:")
            for sub, corr in unique_submissions[:3]:
                print(f"  - {sub} (weight: 0.20)")
            print(f"  - {top_submissions[0]} (weight: 0.40)")  # Give highest weight to best submission
    else:
        print("No high quality submissions found for ensembling.")
    
    print("\nFiltering complete!")
    return results

if __name__ == "__main__":
    filter_submissions()
