import warnings
warnings.filterwarnings("ignore")

from genetic_feature_engineering import GeneticFeatureEngineering
from data_pipeline import create_complete_data_pipeline
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import time

def silent_evaluate(self, chromosome):
    """Fixed evaluation function with better error handling"""
    X_train_subset = self.X_train.copy()
    X_val_subset = self.X_val.copy()
    
    feature_set = []
    
    for feature in chromosome:
        if isinstance(feature, str):
            if feature in self.base_features:
                feature_set.append(feature)
        elif isinstance(feature, tuple):
            operation, operands = feature[0], feature[1:]
            feature_name = self._generate_feature_name(operation, operands)
            X_train_subset[feature_name] = self._apply_transformation(operation, operands, X_train_subset)
            X_val_subset[feature_name] = self._apply_transformation(operation, operands, X_val_subset)
            feature_set.append(feature_name)
            self.feature_transformers[feature_name] = (operation, operands)
    
    if len(feature_set) < 3:
        return 0.0
    
    # Handle missing values
    for col in feature_set:
        if X_train_subset[col].isna().any():
            X_train_subset[col] = X_train_subset[col].fillna(X_train_subset[col].median())
        if X_val_subset[col].isna().any():
            X_val_subset[col] = X_val_subset[col].fillna(X_val_subset[col].median())
    
    try:
        # Use RandomForestRegressor directly
        model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            n_jobs=-1, 
            random_state=42
        )
        model.fit(X_train_subset[feature_set], self.y_train)
        y_pred = model.predict(X_val_subset[feature_set])
        mse = mean_squared_error(self.y_val, y_pred)
        
        # Add more debugging here
        if mse > 1.0:
            return 0.0  # MSE too high, consider it a failed model
            
        fitness = 1 - mse
        return fitness
    except Exception as e:
        # At least print the error for debugging
        print(f"Error in evaluation: {str(e)[:100]}")
        return 0.0

# Custom progress reporter that shows more details
class SimpleProgressReporter:
    def __init__(self, generations):
        self.generations = generations
        self.start_time = None
        
    def start(self):
        self.start_time = time.time()
        print(f"Starting evolution...")
        
    def update(self, generation, best_fitness, best_features, avg_fitness):
        if generation % 5 == 0 or generation == 1:
            print(f"Generation {generation}/{self.generations}: Best={best_fitness:.6f}, Avg={avg_fitness:.6f}, Features={best_features}")
            
    def finish(self):
        elapsed = time.time() - self.start_time
        print(f"Evolution completed in {elapsed:.1f} seconds")

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    X_train_temporal, X_test_temporal = create_complete_data_pipeline()
    y_train = X_train_temporal['rainfall']
    X_train = X_train_temporal.drop(['id', 'rainfall'], axis=1)
    X_test = X_test_temporal.drop(['id'], axis=1)
    
    # Verify data is usable
    print(f"Training data shape: {X_train.shape}")
    print(f"Target shape: {y_train.shape}")
    print(f"Base features: {X_train.columns.tolist()}")
    
    # Try a simple model to verify data works
    print("\nRunning test model...")
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    X_train_small, X_val_small, y_train_small, y_val_small = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    rf.fit(X_train_small, y_train_small)
    val_pred = rf.predict(X_val_small)
    base_mse = mean_squared_error(y_val_small, val_pred)
    print(f"Test model MSE: {base_mse:.6f}, Fitness: {1-base_mse:.6f}")
    
    # Initialize GA with smaller population for faster testing
    ga = GeneticFeatureEngineering(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        population_size=50,  # Smaller for faster testing
        generations=20,      # Fewer generations for testing
        mutation_rate=0.3,
        crossover_rate=0.7
    )
    
    # Override the evaluation function
    ga._evaluate_chromosome = lambda chrom: silent_evaluate(ga, chrom)
    
    # Run evolution
    print("\nStarting genetic algorithm evolution...")
    best_chromosome, best_fitness = ga.evolve()
    
    print(f"\nFinal validation score: {best_fitness:.6f}")
    
    if best_fitness > 0:
        # Create submission only if we have a decent model
        submission, X_train_final, X_test_final, feature_set = ga.create_submission(X_test_temporal)
        
        # Print top features
        print("\nTop features in best chromosome:")
        feature_names = []
        for feature in best_chromosome[:10]:
            if isinstance(feature, str):
                feature_names.append(feature)
            else:
                op, *operands = feature
                name = ga._generate_feature_name(op, operands)
                feature_names.append(name)
        
        for i, name in enumerate(feature_names):
            print(f"{i+1}. {name}")
    else:
        print("No good solution found - fitness is 0")