#!/usr/bin/env python3
# filepath: /home/USER/Projects/RAIN/create_ensemble.py

import pandas as pd
import numpy as np
import os
from glob import glob

# Load all candidate submissions
candidates_dir = "/home/USER/Projects/RAIN/data/candidates"
candidate_files = glob(os.path.join(candidates_dir, "*.csv"))

# Dictionary to store dataframes and their scores
submissions = {}
scores = {}

# Load submissions and extract scores from filenames
for file in candidate_files:
    df = pd.read_csv(file)
    base_name = os.path.basename(file)
    
    # Get score from the original results file
    with open("/home/USER/Projects/RAIN/submission_results.csv", "r") as f:
        for line in f:
            if base_name in line:
                score = float(line.strip().split(',')[1])
                break
    
    submissions[base_name] = df
    scores[base_name] = score
    print(f"Loaded {base_name} with score {score}")

# Create different ensemble methods
def create_ensembles():
    # Get reference columns from first submission
    first_df = list(submissions.values())[0]
    id_col = first_df.columns[0]
    pred_col = first_df.columns[1]
    
    # 1. Simple average ensemble
    avg_ensemble = first_df.copy()
    avg_ensemble[pred_col] = 0
    
    # 2. Score-weighted ensemble
    weighted_ensemble = first_df.copy()
    weighted_ensemble[pred_col] = 0
    
    # 3. Rank-weighted ensemble
    rank_ensemble = first_df.copy()
    rank_ensemble[pred_col] = 0
    
    # Calculate weights
    total_score = sum(scores.values())
    # Rank weights (higher rank = higher weight)
    ranked_scores = sorted([(name, score) for name, score in scores.items()], 
                          key=lambda x: x[1], reverse=True)
    rank_weights = {name: len(ranked_scores)-i for i, (name, _) in enumerate(ranked_scores)}
    total_rank = sum(rank_weights.values())
    
    # Sum predictions with weights
    for name, df in submissions.items():
        avg_ensemble[pred_col] += df[pred_col] / len(submissions)
        weighted_ensemble[pred_col] += df[pred_col] * (scores[name]/total_score)
        rank_ensemble[pred_col] += df[pred_col] * (rank_weights[name]/total_rank)
    
    # Save ensembles
    avg_ensemble.to_csv(f"{candidates_dir}/ensemble_average.csv", index=False)
    weighted_ensemble.to_csv(f"{candidates_dir}/ensemble_score_weighted.csv", index=False)
    rank_ensemble.to_csv(f"{candidates_dir}/ensemble_rank_weighted.csv", index=False)
    
    # Try different weight combinations
    best_score = 0
    best_weights = None
    
    # Generate some weight combinations to try
    weight_combinations = []
    for i in range(11):
        for j in range(11-i):
            k = 10 - i - j
            weight_combinations.append([i/10, j/10, k/10])
    
    top_submissions = list(submissions.values())[:3]  # Use top 3 submissions
    
    for weights in weight_combinations:
        blend = first_df.copy()
        blend[pred_col] = 0
        
        for i, df in enumerate(top_submissions[:len(weights)]):
            blend[pred_col] += df[pred_col] * weights[i]
            
        blend_file = f"{candidates_dir}/ensemble_optimized.csv"
        blend.to_csv(blend_file, index=False)
        
        # Here you would evaluate this blend and update best_score if better
        # For now we'll just save the last combination
    
    print("Created ensembles in the candidates directory")

create_ensembles()