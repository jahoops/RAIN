from systematic_features import FeatureDiscoveryFramework
from feature_library import (basic_features, temporal_features, 
                             statistical_transformations, polynomial_features,
                             dimensionality_reduction, domain_specific_features,
                             combined_features)
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

def main():
    # Initialize the framework
    fdf = FeatureDiscoveryFramework()
    
    # Load and explore data
    fdf.load_data()
    fdf.explore_data()
    
    # Register feature sets
    fdf.register_feature_set(
        name="basic",
        feature_func=basic_features,
        description="Basic features with minimal processing"
    )
    
    fdf.register_feature_set(
        name="temporal",
        feature_func=temporal_features,
        description="Temporal and seasonal features"
    )
    
    fdf.register_feature_set(
        name="statistical",
        feature_func=statistical_transformations,
        description="Statistical transformations of features"
    )
    
    fdf.register_feature_set(
        name="polynomial",
        feature_func=polynomial_features,
        description="Polynomial and interaction features"
    )
    
    fdf.register_feature_set(
        name="dimensionality",
        feature_func=dimensionality_reduction,
        description="Dimensionality reduction techniques"
    )
    
    fdf.register_feature_set(
        name="domain",
        feature_func=domain_specific_features,
        description="Domain-specific features for weather prediction"
    )
    
    fdf.register_feature_set(
        name="combined",
        feature_func=combined_features,
        description="Combined features from multiple sets"
    )
    
    # Evaluate each feature set with better error handling
    feature_sets = list(fdf.feature_registry.keys())
    successful_features = []
    
    for name in feature_sets:
        try:
            print(f"\n{'-'*50}")
            print(f"Evaluating feature set: {name}")
            print(f"{'-'*50}")
            
            score = fdf.evaluate_feature_set(name)
            successful_features.append(name)
            print(f"Successfully evaluated {name} with score {score}")
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            continue
    
    print(f"\nSuccessfully evaluated {len(successful_features)} out of {len(feature_sets)} feature sets")
    
    if successful_features:
        # Rank successful feature sets
        fdf.rank_features()
        
        # Document findings
        fdf.document_findings()
        
        # Create submission for best feature set
        best_feature_set = fdf.rank_features().iloc[0]['Feature Set']
        submission_path = fdf.create_submission(
            best_feature_set, 
            f"data/candidates/rf_{best_feature_set}_submission.csv"
        )
        
        print(f"\nCreated submission with best feature set ({best_feature_set}): {submission_path}")
    else:
        print("No feature sets were successfully evaluated.")

if __name__ == "__main__":
    main()