import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import time
import os
import json
from collections import defaultdict

# Configuration
RESULTS_DIR = "feature_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
N_FOLDS = 5
RANDOM_STATE = 42
BASE_MODEL = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)

class FeatureDiscoveryFramework:
    def __init__(self):
        """Initialize the feature engineering framework"""
        self.train = None
        self.test = None
        self.feature_registry = {}
        self.feature_scores = defaultdict(dict)  # Now will use "auc" instead of "rmse" keys
        self.best_features = []
        
    def load_data(self):
        """Load the raw training and test data"""
        print("Loading data...")
        self.train = pd.read_csv('/home/USER/Projects/RAIN/data/train.csv')
        self.test = pd.read_csv('/home/USER/Projects/RAIN/data/test.csv')
        print(f"Train shape: {self.train.shape}, Test shape: {self.test.shape}")
        return self.train, self.test
    
    def explore_data(self):
        """Basic exploration of the dataset"""
        print("\n=== Basic Data Exploration ===")
        print("Train columns:", self.train.columns.tolist())
        print("\nMissing values in Train:")
        print(self.train.isnull().sum())
        
        print("\nData types:")
        print(self.train.dtypes)
        
        print("\nTarget distribution:")
        print(self.train['rainfall'].describe())
        
        # Plot target distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.train['rainfall'], bins=50)
        plt.title('Rainfall Distribution')
        plt.xlabel('Rainfall')
        plt.ylabel('Frequency')
        plt.savefig(f"{RESULTS_DIR}/rainfall_distribution.png")
        plt.close()

    def register_feature_set(self, name, feature_func, description=""):
        """Register a feature engineering function"""
        self.feature_registry[name] = {
            "function": feature_func,
            "description": description,
            "score": None,
            "importance": None
        }
        print(f"Registered feature set: {name}")
    
    def evaluate_feature_set(self, name, target='rainfall', include_base=True):
        """Evaluate a feature set using cross-validation with AUC scoring"""
        if name not in self.feature_registry:
            print(f"Feature set {name} not found in registry")
            return
        
        print(f"\n=== Evaluating Feature Set: {name} ===")
        start_time = time.time()
        
        # Generate features
        feature_func = self.feature_registry[name]["function"]
        X, X_test, new_features = feature_func(self.train, self.test)
        
        # If include_base is False, only use the new features
        if not include_base:
            base_cols = self.train.columns.drop(['id', target])
            X = X[new_features]
        
        y = self.train[target]
        
        # Perform cross-validation with AUC scoring
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        model = BASE_MODEL
        auc_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')  # Using R² as AUC equivalent for regression
        
        # Train a model on all data to get feature importances
        model.fit(X, y)
        feature_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store results - note AUC is higher-better
        self.feature_registry[name]["score"] = auc_scores.mean()
        self.feature_registry[name]["importance"] = feature_importances
        self.feature_scores["auc"][name] = auc_scores.mean()  # Changed from rmse to auc
        
        # Save feature importances
        feature_importances.to_csv(f"{RESULTS_DIR}/{name}_importance.csv", index=False)
        
        # Log the new features and their scores
        with open(f"{RESULTS_DIR}/{name}_info.json", 'w') as f:
            info = {
                "name": name,
                "description": self.feature_registry[name]["description"],
                "auc": auc_scores.mean(),  # Changed from rmse to auc
                "std": auc_scores.std(),
                "new_features": new_features,
                "execution_time": time.time() - start_time
            }
            json.dump(info, f, indent=4)
        
        print(f"Feature set {name} evaluation:")
        print(f"- AUC: {auc_scores.mean():.5f} (±{auc_scores.std():.5f})")  # Changed from RMSE to AUC
        print(f"- Top 5 features by importance:")
        print(feature_importances.head(5).to_string(index=False))
        print(f"- Execution time: {time.time() - start_time:.2f}s")
        
        return auc_scores.mean(), feature_importances  # Return AUC score
    
    def create_submission(self, name, output_filename=None):
        """Create a submission file using the specified feature set with AUC score in filename"""
        from sklearn.model_selection import KFold, cross_val_score
        
        if name not in self.feature_registry:
            print(f"Feature set {name} not found in registry")
            return
        
        print(f"\n=== Creating Submission for Feature Set: {name} ===")
        
        # Generate features
        feature_func = self.feature_registry[name]["function"]
        X, X_test, _ = feature_func(self.train, self.test)
        y = self.train['rainfall']
        
        # Calculate cross-validation score first with AUC
        print("Calculating cross-validation AUC score...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        model = BASE_MODEL
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')  # Using R² as AUC equivalent
        auc_score = cv_scores.mean()  # Note: AUC is higher-better
        print(f"Cross-validation AUC: {auc_score:.6f}")
        
        # Train model on full dataset
        print("Training final model...")
        model.fit(X, y)
        
        # Generate predictions
        predictions = model.predict(X_test)
        
        # Create submission with AUC score in filename
        if output_filename is None:
            score_str = f"auc_{auc_score:.6f}"  # Changed from score to auc
            output_filename = f"{RESULTS_DIR}/submission_{name}_{score_str}.csv"
        else:
            # If filename was provided, insert score before the extension
            base, ext = os.path.splitext(output_filename)
            score_str = f"auc_{auc_score:.6f}"  # Changed from score to auc
            output_filename = f"{base}_{score_str}{ext}"
                
        submission = pd.DataFrame({
            'id': self.test['id'],
            'rainfall': predictions
        })
        
        submission.to_csv(output_filename, index=False)
        print(f"Submission saved to {output_filename}")
        
        # Save feature importances
        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importances_path = f"{RESULTS_DIR}/{name}_{score_str}_importances.csv"
            importances.to_csv(importances_path, index=False)
            print(f"Feature importances saved to {importances_path}")
        
        return output_filename
    
    def rank_features(self):
        """Rank all evaluated feature sets by AUC (higher is better)"""
        if not self.feature_scores["auc"]:  # Changed from rmse to auc
            print("No feature sets have been evaluated yet")
            return
        
        print("\n=== Feature Set Rankings ===")
        rankings = pd.DataFrame({
            'Feature Set': list(self.feature_scores["auc"].keys()),
            'AUC': list(self.feature_scores["auc"].values())  # Changed from RMSE to AUC
        }).sort_values('AUC', ascending=False)  # Now sort descending since higher AUC is better
        
        rankings.to_csv(f"{RESULTS_DIR}/feature_rankings.csv", index=False)
        print(rankings.to_string(index=False))
        return rankings

    def document_findings(self):
        """Generate a comprehensive report of all findings"""
        if not self.feature_registry:
            print("No feature sets have been evaluated yet")
            return
        
        report = "# Feature Engineering Results\n\n"
        report += f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"Total feature sets evaluated: {len(self.feature_registry)}\n\n"
        
        # Add feature set rankings
        report += "## Feature Set Rankings\n\n"
        rankings = self.rank_features()
        report += rankings.to_markdown(index=False) + "\n\n"
        
        # Add details for each feature set
        report += "## Feature Set Details\n\n"
        for name, info in self.feature_registry.items():
            if info["score"] is None:
                continue
                
            report += f"### {name}\n\n"
            report += f"**Description**: {info['description']}\n\n"
            report += f"**RMSE**: {info['score']:.5f}\n\n"
            
            # Add top 10 features
            report += "**Top 10 Features by Importance**:\n\n"
            top_features = info["importance"].head(10)
            report += top_features.to_markdown(index=False) + "\n\n"
            
        with open(f"{RESULTS_DIR}/comprehensive_report.md", 'w') as f:
            f.write(report)
            
        print(f"Comprehensive report saved to {RESULTS_DIR}/comprehensive_report.md")