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
from copy import deepcopy

# Fixed missing values function to avoid the Series truth value error
def fixed_fill_missing_values(df):
    """Fill missing values in a DataFrame"""
    df_copy = df.copy()
    for col in df_copy.columns:
        # Check for NaN values - proper way to handle this
        missing_count = df_copy[col].isna().sum()
        if missing_count > 0:
            if np.issubdtype(df_copy[col].dtype, np.number):
                # For numeric, use median
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            else:
                # For categorical, use mode with safeguards
                mode_vals = df_copy[col].mode()
                if len(mode_vals) > 0:
                    df_copy[col] = df_copy[col].fillna(mode_vals[0])
                else:
                    df_copy[col] = df_copy[col].fillna("unknown")
    return df_copy

# Fixed apply_to_test_data function
def fixed_apply_to_test(ga):
    """Apply the best feature set to test data - fixed version"""
    print("\nApplying best feature set to test data (fixed version)...")
    
    # Create feature matrices
    X_train_final = ga.X_train.copy()
    X_test_final = ga.X_test.copy()
    
    # Apply transformations from the best chromosome
    feature_set = []
    
    for feature in ga.best_chromosome:
        try:
            if isinstance(feature, str):
                # Base feature
                if feature in ga.base_features:
                    feature_set.append(feature)
            else:
                # Derived feature
                operation, operands = feature[0], feature[1:]
                feature_name = ga._generate_feature_name(operation, operands)
                
                # Apply to train - with error handling
                try:
                    X_train_final[feature_name] = ga._apply_transformation(
                        operation, operands, X_train_final)
                    
                    # Apply to test
                    X_test_final[feature_name] = ga._apply_transformation(
                        operation, operands, X_test_final)
                    
                    feature_set.append(feature_name)
                except Exception as e:
                    print(f"Error applying feature {feature_name}: {str(e)[:80]}")
        except Exception as e:
            print(f"Error processing feature: {str(e)[:80]}")
    
    # Filter out features not in X_test_final
    feature_set = [f for f in feature_set if f in X_train_final.columns and f in X_test_final.columns]
    
    print(f"Using {len(feature_set)} features for final model")
    
    # Fill missing values using our fixed function
    X_train_features = fixed_fill_missing_values(X_train_final[feature_set])
    X_test_features = fixed_fill_missing_values(X_test_final[feature_set])
    
    return X_train_features, X_test_features, feature_set

def silent_evaluate(ga, chromosome):
    """Fixed evaluation function with better error handling"""
    X_train_subset = ga.X_train.copy()
    X_val_subset = ga.X_val.copy()
    
    feature_set = []
    
    for feature in chromosome:
        try:
            if isinstance(feature, str):
                if feature in ga.base_features:
                    feature_set.append(feature)
            elif isinstance(feature, tuple):
                operation, operands = feature[0], feature[1:]
                feature_name = ga._generate_feature_name(operation, operands)
                
                # Apply transformations with error handling
                try:
                    X_train_subset[feature_name] = ga._apply_transformation(
                        operation, operands, X_train_subset)
                    X_val_subset[feature_name] = ga._apply_transformation(
                        operation, operands, X_val_subset)
                    feature_set.append(feature_name)
                    ga.feature_transformers[feature_name] = (operation, operands)
                except Exception as e:
                    # Skip this feature if transformation fails
                    continue
        except Exception as e:
            continue
    
    if len(feature_set) < 3:
        return 0.0
    
    # Handle missing values more carefully
    for col in feature_set:
        if col in X_train_subset.columns and col in X_val_subset.columns:
            # Check for NaN values
            if X_train_subset[col].isna().sum() > 0:
                X_train_subset[col] = X_train_subset[col].fillna(X_train_subset[col].median())
            if X_val_subset[col].isna().sum() > 0:
                X_val_subset[col] = X_val_subset[col].fillna(X_val_subset[col].median())
    
    try:
        # Use RandomForestRegressor directly
        model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            n_jobs=-1, 
            random_state=42
        )
        model.fit(X_train_subset[feature_set], ga.y_train)
        y_pred = model.predict(X_val_subset[feature_set])
        mse = mean_squared_error(ga.y_val, y_pred)
        
        # Add more debugging here
        if mse > 1.0:
            print(f"  Feature set with {len(feature_set)} features has high MSE: {mse:.6f}")
            return 0.0
        else:
            print(f"  Feature set with {len(feature_set)} features has MSE: {mse:.6f}, Fitness: {1-mse:.6f}")
            
        fitness = 1 - mse
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
    
    # Verify data is usable
    print(f"Training data shape: {X_train.shape}")
    print(f"Target shape: {y_train.shape}")
    print(f"Base features: {X_train.columns.tolist()}")
    
    # Try a simple model to verify data works
    print("\nRunning test model...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
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
        population_size=30,  # Even smaller for faster testing
        generations=15,      # Fewer generations for testing
        mutation_rate=0.3,
        crossover_rate=0.7
    )
    
    # Override the evaluation function
    ga._evaluate_chromosome = lambda chrom: silent_evaluate(ga, chrom)
    
    # Run evolution with progress tracking
    print("\nStarting genetic algorithm evolution...")
    start_time = time.time()
    
    # We'll track progress manually
    for gen in range(ga.generations):
        if gen == 0:
            # Initialize
            ga._initialize_population()
            ga.fitness_scores = [ga._evaluate_chromosome(chrom) for chrom in ga.population]
            best_idx = np.argmax(ga.fitness_scores)
            ga.best_chromosome = deepcopy(ga.population[best_idx])
            ga.best_fitness = ga.fitness_scores[best_idx]
            
        # Create new population
        new_population = [deepcopy(ga.best_chromosome)]  # Elitism
        while len(new_population) < ga.population_size:
            # Select parents
            parent1 = ga.population[np.random.randint(0, len(ga.population))]
            parent2 = ga.population[np.random.randint(0, len(ga.population))]
            
            # Crossover
            if np.random.random() < ga.crossover_rate:
                point = np.random.randint(1, min(len(parent1), len(parent2)) - 1)
                child1 = parent1[:point] + parent2[point:]
                child2 = parent2[:point] + parent1[point:]
            else:
                child1, child2 = deepcopy(parent1), deepcopy(parent2)
                
            # Mutation
            if np.random.random() < ga.mutation_rate:
                # Simple mutation: add or remove a feature
                if len(child1) > 3 and np.random.random() < 0.5:
                    idx = np.random.randint(0, len(child1))
                    child1.pop(idx)
                else:
                    if np.random.random() < 0.5:
                        # Add a base feature
                        avail_features = [f for f in ga.base_features if f not in child1]
                        if avail_features:
                            child1.append(np.random.choice(avail_features))
            
            # Add to new population
            new_population.append(child1)
            if len(new_population) < ga.population_size:
                new_population.append(child2)
        
        # Update population
        ga.population = new_population
        
        # Evaluate new population
        ga.fitness_scores = [ga._evaluate_chromosome(chrom) for chrom in ga.population]
        
        # Update best chromosome
        best_idx = np.argmax(ga.fitness_scores)
        if ga.fitness_scores[best_idx] > ga.best_fitness:
            ga.best_chromosome = deepcopy(ga.population[best_idx])
            ga.best_fitness = ga.fitness_scores[best_idx]
            
        # Report progress
        avg_fitness = np.mean(ga.fitness_scores)
        print(f"Generation {gen+1}/{ga.generations}: Best={ga.best_fitness:.6f}, Avg={avg_fitness:.6f}, Features={len(ga.best_chromosome)}")
    
    elapsed = time.time() - start_time
    print(f"Evolution completed in {elapsed:.1f} seconds")
    
    print(f"\nFinal validation score: {ga.best_fitness:.6f}")
    
    if ga.best_fitness > 0:
        # Create submission using our fixed function
        X_train_final, X_test_final, feature_set = fixed_apply_to_test(ga)
        
        # Train final model
        print("Training final model...")
        final_model = RandomForestRegressor(
            n_estimators=1000,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=87
        )
        final_model.fit(X_train_final, ga.y_train)
        
        # Make predictions
        predictions = final_model.predict(X_test_final)
        
        # Create submission
        submission = pd.DataFrame({
            'id': X_test_temporal['id'],
            'rainfall': predictions
        })
        
        # Save the submission
        submission.to_csv('/home/USER/Projects/RAIN/data/candidates/fixed_genetic_features.csv', index=False)
        print("Created submission with genetically engineered features")
        
        # Print top features
        print("\nTop features in best chromosome:")
        feature_names = []
        for feature in ga.best_chromosome[:10]:
            if isinstance(feature, str):
                feature_names.append(feature)
            else:
                op, *operands = feature
                try:
                    name = ga._generate_feature_name(op, operands)
                    feature_names.append(name)
                except:
                    feature_names.append(f"{op}_{operands}")
        
        for i, name in enumerate(feature_names):
            print(f"{i+1}. {name}")
    else:
        print("No good solution found - fitness is 0")