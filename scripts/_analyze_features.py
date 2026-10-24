import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import os

# Find all model pickle files associated with the candidates
model_files = glob.glob("/home/USER/Projects/RAIN/models/**/*.pkl", recursive=True)

# Dictionary to store feature importance
feature_importance = {}

# Load each model and extract feature importance if available
for model_file in model_files:
    try:
        model = pickle.load(open(model_file, 'rb'))
        
        # Extract feature importance based on model type
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
            feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [f"feature_{i}" for i in range(len(importances))]
            
            for feature, importance in zip(feature_names, importances):
                if feature not in feature_importance:
                    feature_importance[feature] = []
                feature_importance[feature].append(importance)
                
        elif hasattr(model, 'coef_'):
            # For linear models
            importances = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [f"feature_{i}" for i in range(len(importances))]
            
            for feature, importance in zip(feature_names, importances):
                if feature not in feature_importance:
                    feature_importance[feature] = []
                feature_importance[feature].append(abs(importance))
                
        print(f"Extracted features from {os.path.basename(model_file)}")
    except Exception as e:
        print(f"Could not process {model_file}: {e}")

# Aggregate importance across models
if feature_importance:
    avg_importance = {feature: np.mean(importances) for feature, importances in feature_importance.items()}
    
    # Sort features by importance
    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 most important features across models:")
    for feature, importance in sorted_features[:10]:
        print(f"{feature}: {importance:.4f}")
        
    # Plot top features
    top_features = sorted_features[:15]
    plt.figure(figsize=(10, 8))
    plt.barh([f[0] for f in top_features], [f[1] for f in top_features])
    plt.title("Top 15 Most Important Features")
    plt.xlabel("Average Importance")
    plt.tight_layout()
    plt.savefig("/home/USER/Projects/RAIN/data/feature_importance.png")
    print("\nFeature importance plot saved to /home/USER/Projects/RAIN/data/feature_importance.png")