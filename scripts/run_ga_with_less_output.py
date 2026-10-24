import warnings
# Suppress all warnings 
warnings.filterwarnings("ignore")

from genetic_feature_engineering import GeneticFeatureEngineering
from data_pipeline import create_complete_data_pipeline
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Override the evaluate_chromosome method to be silent
def silent_evaluate(self, chromosome):
    # Same as the original but without prints
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
    
    # Handle missing values silently
    for col in feature_set:
        if X_train_subset[col].isna().any():
            X_train_subset[col] = X_train_subset[col].fillna(X_train_subset[col].median())
        if X_val_subset[col].isna().any():
            X_val_subset[col] = X_val_subset[col].fillna(X_val_subset[col].median())
    
    try:
        model = self.model_cls(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
        model.fit(X_train_subset[feature_set], self.y_train)
        y_pred = model.predict(X_val_subset[feature_set])
        mse = mean_squared_error(self.y_val, y_pred)
        fitness = 1 - mse
        if fitness > 0.95:
            fitness = 0.95
        return fitness
    except Exception as e:
        return 0.0

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    X_train_temporal, X_test_temporal = create_complete_data_pipeline()
    y_train = X_train_temporal['rainfall']
    X_train = X_train_temporal.drop(['id', 'rainfall'], axis=1)
    X_test = X_test_temporal.drop(['id'], axis=1)
    
    # Initialize GA
    ga = GeneticFeatureEngineering(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        population_size=100,
        generations=50,
        mutation_rate=0.3,
        crossover_rate=0.7
    )
    
    # Override the evaluate_chromosome method to be silent
    ga._evaluate_chromosome = lambda chrom: silent_evaluate(ga, chrom)
    
    # Run evolution
    print("\nStarting genetic algorithm evolution...")
    best_chromosome, best_fitness = ga.evolve()
    
    # Create submission
    submission, X_train_final, X_test_final, feature_set = ga.create_submission(X_test_temporal)
    
    print(f"\nFinal validation score: {best_fitness:.6f}")
    
    # Print top 10 features
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