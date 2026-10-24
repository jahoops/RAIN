import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time
import os
import pickle
from copy import deepcopy
from scipy.stats import skew, kurtosis
from joblib import Parallel, delayed
import warnings
from tqdm.auto import tqdm
import sys

# Suppress specific RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                       message="Precision loss occurred in moment calculation")

# Suppress ALL warnings related to the calculations
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Create a custom progress reporter
class EvolutionProgressReporter:
    def __init__(self, total_generations):
        self.total_generations = total_generations
        # No progress bar, we'll print manually once per generation
        self.best_fitness = 0
        self.best_features = 0
        self.start_time = time.time()
        
    def update(self, generation, best_fitness, best_features, avg_fitness):
        self.best_fitness = max(self.best_fitness, best_fitness)
        self.best_features = best_features
        
        # Calculate elapsed time and ETA
        elapsed = time.time() - self.start_time
        if generation > 0:  # Avoid division by zero
            eta = (elapsed / generation) * (self.total_generations - generation)
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
        else:
            eta_str = "N/A"
            
        # Build a progress bar manually
        width = 30
        filled = int(width * generation / self.total_generations)
        progress_bar = "█" * filled + "░" * (width - filled)
        
        # Print progress info once per generation
        print(f"\rGeneration {generation}/{self.total_generations} [{progress_bar}] " +
              f"Best: {best_fitness:.6f}, Avg: {avg_fitness:.6f}, Features: {best_features}, " +
              f"ETA: {eta_str}", end="")
        
        # Every 10 generations, add a newline for better readability
        if generation % 10 == 0 and generation > 0:
            print()

    def close(self):
        print("\nEvolution complete!")

class GeneticFeatureEngineering:
    def __init__(self, X_train, y_train, X_test, test_size=0.2, population_size=50, 
                 generations=30, mutation_rate=0.2, crossover_rate=0.7):
        # Split data
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train, y_train, test_size=test_size, random_state=42)
        self.X_test = X_test
        self.test_ids = X_test.index if hasattr(X_test, 'index') else None
        
        # Base features and parameters
        self.base_features = list(X_train.columns)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Tracking
        self.population = []
        self.best_chromosome = None
        self.best_fitness = 0.0
        self.fitness_history = []
        self.feature_transformers = {}
        self.generation_summaries = []
        
        # Create output directory
        os.makedirs('genetic_engineering', exist_ok=True)
        
        print(f"Initialized genetic algorithm with {len(self.base_features)} base features")
        
    def _evaluate_chromosome(self, chromosome):
        """Evaluate a chromosome (feature set) using validation score"""
        # Create feature matrix
        X_train_subset = self.X_train.copy()
        X_val_subset = self.X_val.copy()
        
        # Apply transformations defined in chromosome
        feature_set = []
        
        for feature in chromosome:
            # Handle base features (strings)
            if isinstance(feature, str):
                if feature in self.base_features:
                    feature_set.append(feature)
            # Handle complex features (tuples)
            elif isinstance(feature, tuple):
                operation, operands = feature[0], feature[1:]
                feature_name = self._generate_feature_name(operation, operands)
                
                # Apply the transformation to training data
                X_train_subset[feature_name] = self._apply_transformation(
                    operation, operands, X_train_subset)
                
                # Apply the same transformation to validation data
                X_val_subset[feature_name] = self._apply_transformation(
                    operation, operands, X_val_subset)
                
                feature_set.append(feature_name)
                
                # Store the transformer for later use with test data
                self.feature_transformers[feature_name] = (operation, operands)
        
        # Check if we have enough features
        if len(feature_set) < 3:
            return 0.0  # Penalty for too few features
        
        # Handle missing values
        for col in feature_set:
            if X_train_subset[col].isna().any():
                X_train_subset[col] = X_train_subset[col].fillna(X_train_subset[col].median())
            if X_val_subset[col].isna().any():
                X_val_subset[col] = X_val_subset[col].fillna(X_val_subset[col].median())
        
        try:
            # Train a model
            model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10,
                n_jobs=-1, 
                random_state=42
            )
            
            model.fit(X_train_subset[feature_set], self.y_train)
            y_pred = model.predict(X_val_subset[feature_set])
            
            # Calculate fitness (1 - MSE)
            mse = mean_squared_error(self.y_val, y_pred)
            
            # Add debugging for suspicious values
            if mse < 0.0001:  # If MSE is suspiciously low
                print(f"WARNING: Extremely small MSE: {mse:.10f}")
                print(f"Feature set: {feature_set}")
            
            fitness = 1 - mse
            
            # Cap fitness at realistic values
            if fitness > 0.95:
                print(f"Capping suspiciously high fitness: {fitness:.6f}")
                fitness = 0.95
                
            return fitness
        except Exception as e:
            print(f"Error evaluating chromosome: {e}")
            return 0.0
    
    def _generate_feature_name(self, operation, operands):
        """Generate a descriptive name for a feature"""
        if operation == 'log':
            return f'log_{operands[0]}'
        elif operation == 'sqrt':
            return f'sqrt_{operands[0]}'
        elif operation == 'square':
            return f'square_{operands[0]}'
        elif operation == 'cube':
            return f'cube_{operands[0]}'
        elif operation == 'inverse':
            return f'inverse_{operands[0]}'
        elif operation == 'multiply':
            return f'{operands[0]}_x_{operands[1]}'
        elif operation == 'divide':
            return f'{operands[0]}_div_{operands[1]}'
        elif operation == 'add':
            return f'{operands[0]}_plus_{operands[1]}'
        elif operation == 'subtract':
            return f'{operands[0]}_minus_{operands[1]}'
        elif operation == 'lag':
            return f'{operands[0]}_lag{operands[1]}'
        elif operation == 'diff':
            return f'{operands[0]}_diff'
        elif operation == 'rolling_mean':
            return f'{operands[0]}_roll{operands[1]}_mean'
        elif operation == 'rolling_std':
            return f'{operands[0]}_roll{operands[1]}_std'
        elif operation == 'rolling_max':
            return f'{operands[0]}_roll{operands[1]}_max'
        elif operation == 'rolling_min':
            return f'{operands[0]}_roll{operands[1]}_min'
        elif operation == 'rolling_skew':
            return f'{operands[0]}_roll{operands[1]}_skew'
        elif operation == 'rolling_kurt':
            return f'{operands[0]}_roll{operands[1]}_kurt'
        elif operation == 'cubic_root':
            return f'cbrt_{operands[0]}'
        elif operation == 'exp':
            return f'exp_{operands[0]}'
        elif operation == 'sigmoid':
            return f'sigmoid_{operands[0]}'
        elif operation == 'rank':
            return f'rank_{operands[0]}'
        else:
            return f'{operation}_{"_".join(map(str, operands))}'
    
    def _apply_transformation(self, operation, operands, data):
        """Apply a transformation to data"""
        try:
            if operation == 'log':
                # Log transform with safeguard for non-positive values
                x = data[operands[0]]
                min_val = x.min()
                if (x < 0).any():
                    return np.sqrt(x - x.min() + 0.01)
                return np.log(x)
                
            elif operation == 'sqrt':
                # Square root with safeguard for negative values
                x = data[operands[0]]
                if (x < 0).any():
                    return np.sqrt(x - x.min() + 0.01)
                return np.sqrt(x)
                
            elif operation == 'square':
                return data[operands[0]] ** 2
                
            elif operation == 'cube':
                return data[operands[0]] ** 3
                
            elif operation == 'inverse':
                # Inverse with safeguard for zeros
                x = data[operands[0]]
                if (x.abs() < 1e-10).any():
                    return 1 / (x + 1e-10)
                return 1 / x
                
            elif operation == 'multiply':
                return data[operands[0]] * data[operands[1]]
                
            elif operation == 'divide':
                # Division with safeguard for zeros
                denominator = data[operands[1]]
                if (denominator.abs() < 1e-10).any():
                    return data[operands[0]] / (denominator + 1e-10)
                return data[operands[0]] / denominator
                
            elif operation == 'add':
                return data[operands[0]] + data[operands[1]]
                
            elif operation == 'subtract':
                return data[operands[0]] - data[operands[1]]
                
            elif operation == 'lag':
                # Create a lag feature
                return data[operands[0]].shift(operands[1])
                
            elif operation == 'diff':
                # Difference feature
                return data[operands[0]].diff()
                
            elif operation == 'rolling_mean':
                # Rolling mean
                window = operands[1]
                return data[operands[0]].rolling(window=window).mean()
                
            elif operation == 'rolling_std':
                # Rolling standard deviation
                window = operands[1]
                return data[operands[0]].rolling(window=window).std()
                
            elif operation == 'rolling_max':
                # Rolling maximum
                window = operands[1]
                return data[operands[0]].rolling(window=window).max()
                
            elif operation == 'rolling_min':
                # Rolling minimum
                window = operands[1]
                return data[operands[0]].rolling(window=window).min()
                
            elif operation == 'rolling_skew':
                # Rolling skewness with error handling
                window = operands[1]
                try:
                    # Use with_min_periods to ensure we have enough data
                    result = data[operands[0]].rolling(window=window, min_periods=3).apply(
                        lambda x: skew(x) if len(x) >= 3 and x.std() > 1e-6 else np.nan
                    )
                    # Fill NaN values with 0 (neutral skew)
                    result = result.fillna(0)
                    return result
                except Exception as e:
                    print(f"Error in rolling_skew: {e}")
                    return pd.Series(0, index=data.index)  # Return neutral skew
                
            elif operation == 'rolling_kurt':
                # Rolling kurtosis with error handling
                window = operands[1]
                try:
                    # Use with_min_periods and check for valid data
                    result = data[operands[0]].rolling(window=window, min_periods=5).apply(
                        lambda x: kurtosis(x) if len(x) >= 5 and x.std() > 1e-6 else np.nan
                    )
                    # Fill NaN values with 0 (mesokurtic distribution)
                    result = result.fillna(3)  # Normal distribution has kurtosis of 3
                    return result
                except Exception as e:
                    # Silently handle errors
                    return pd.Series(3, index=data.index)  # Return neutral kurtosis
                
            elif operation == 'cubic_root':
                return np.cbrt(data[operands[0]])
                
            elif operation == 'exp':
                # Safe exponential (clip values to avoid overflow)
                return np.exp(np.clip(data[operands[0]], -10, 10))
                
            elif operation == 'sigmoid':
                # Sigmoid transformation: 1/(1+exp(-x))
                return 1 / (1 + np.exp(-data[operands[0]]))
                
            elif operation == 'rank':
                # Percentile rank transformation
                return data[operands[0]].rank(pct=True)
                
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
        except Exception as e:
            print(f"Error applying transformation {operation} to {operands}: {e}")
            # Return NaN series as fallback
            return pd.Series(np.nan, index=data.index)
    
    def _initialize_population(self):
        """Initialize the population with random chromosomes"""
        print("Initializing population...")
        
        self.population = []
        
        # Add a base chromosome with only original features
        base_chromosome = random.sample(self.base_features, 
                                        k=min(10, len(self.base_features)))
        self.population.append(base_chromosome)
        
        # Add a simple chromosome with most basic features
        simple_features = [f for f in self.base_features if any(name in f for name in 
                      ['rainfall', 'humidity', 'pressure', 'cloud', 'temparature'])]
        if len(simple_features) >= 3:
            self.population.append(simple_features[:min(8, len(simple_features))])
        
        # Create random chromosomes
        for _ in range(self.population_size - len(self.population)):
            # Start with some base features
            chromosome = random.sample(
                self.base_features, 
                k=random.randint(3, min(10, len(self.base_features)))
            )
            
            # Add some derived features
            num_derived = random.randint(2, 10)
            for _ in range(num_derived):
                derived_feature = self._create_random_feature()
                if derived_feature:
                    chromosome.append(derived_feature)
            
            self.population.append(chromosome)
        
        print(f"Created initial population of {len(self.population)} chromosomes")
    
    def _create_random_feature(self):
        """Create a random derived feature"""
        # Select operation type
        op_type = random.choice([
            'unary',     # Operations on a single feature
            'binary',    # Operations between two features
            'temporal',  # Time-related operations
            'window'     # Window-based operations
        ])
        
        try:
            if op_type == 'unary':
                operation = random.choice(['log', 'sqrt', 'square', 'cube', 'inverse', 'cubic_root',
                                           'exp', 'sigmoid', 'rank'])  # Added new transformations
                feature = random.choice(self.base_features)
                return (operation, feature)
                
            elif op_type == 'binary':
                operation = random.choice(['multiply', 'divide', 'add', 'subtract'])
                feature1 = random.choice(self.base_features)
                feature2 = random.choice([f for f in self.base_features if f != feature1])
                return (operation, feature1, feature2)
                
            elif op_type == 'temporal':
                # Only create lag/diff features for rainfall-related columns
                rain_features = [f for f in self.base_features if 'rain' in f]
                
                if not rain_features:
                    return None
                    
                operation = random.choice(['lag', 'diff'])
                feature = random.choice(rain_features)
                
                if operation == 'lag':
                    lag = random.randint(1, 7)
                    return (operation, feature, lag)
                else:
                    return (operation, feature)
                    
            elif op_type == 'window':
                # Window operations for rainfall or meteorological features
                rain_features = [f for f in self.base_features if 'rain' in f]
                weather_features = [f for f in self.base_features 
                                   if any(w in f for w in ['temp', 'humidity', 'cloud', 'pressure'])]
                
                if rain_features:
                    feature = random.choice(rain_features)
                elif weather_features:
                    feature = random.choice(weather_features)
                else:
                    return None
                
                operation = random.choice([
                    'rolling_mean', 'rolling_std', 'rolling_max', 
                    'rolling_min', 'rolling_skew', 'rolling_kurt'
                ])
                
                window = random.choice([3, 5, 7, 10, 14])
                return (operation, feature, window)
        
        except Exception as e:
            print(f"Error creating random feature: {e}")
            return None
    
    def _select_parents(self):
        """Select parents using tournament selection"""
        tournament_size = 3
        parent1_idx = random.randint(0, len(self.population) - 1)
        
        # Tournament for first parent
        for _ in range(tournament_size - 1):
            idx = random.randint(0, len(self.population) - 1)
            if self.fitness_scores[idx] > self.fitness_scores[parent1_idx]:
                parent1_idx = idx
        
        # Tournament for second parent
        parent2_idx = random.randint(0, len(self.population) - 1)
        for _ in range(tournament_size - 1):
            idx = random.randint(0, len(self.population) - 1)
            if self.fitness_scores[idx] > self.fitness_scores[parent2_idx]:
                parent2_idx = idx
                
        return self.population[parent1_idx], self.population[parent2_idx]
    
    def _crossover(self, parent1, parent2):
        """Perform crossover between two parents"""
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
            
        # Point for crossover
        point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        
        # Create children
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        
        return child1, child2
    
    def _mutate(self, chromosome):
        """Mutate a chromosome"""
        if random.random() > self.mutation_rate:
            return chromosome
            
        # Choose mutation type
        mutation_type = random.choice([
            'add_feature',
            'remove_feature',
            'replace_feature',
            'add_derived_feature'
        ])
        
        if mutation_type == 'add_feature' and len(self.base_features) > 0:
            # Add a base feature not already in chromosome
            existing = set([f for f in chromosome if isinstance(f, str)])
            available = [f for f in self.base_features if f not in existing]
            
            if available:
                chromosome.append(random.choice(available))
                
        elif mutation_type == 'remove_feature' and len(chromosome) > 3:
            # Remove a random feature
            idx = random.randint(0, len(chromosome) - 1)
            chromosome.pop(idx)
            
        elif mutation_type == 'replace_feature':
            # Replace a feature with another base feature
            idx = random.randint(0, len(chromosome) - 1)
            if isinstance(chromosome[idx], str):  # Only replace base features
                existing = set([f for f in chromosome if isinstance(f, str)])
                available = [f for f in self.base_features if f not in existing]
                
                if available:
                    chromosome[idx] = random.choice(available)
                    
        elif mutation_type == 'add_derived_feature':
            # Add a new derived feature
            derived_feature = self._create_random_feature()
            if derived_feature:
                chromosome.append(derived_feature)
                
        return chromosome
    
    def _fill_missing_values(self, df):
        """Fill missing values in a DataFrame"""
        for col in df.columns:
            if df[col].isna().any():
                if df[col].dtype in [np.float64, np.int64]:
                    # For numeric, use median
                    df[col] = df[col].fillna(df[col].median())
                else:
                    # For categorical, use mode
                    df[col] = df[col].fillna(df[col].mode().iloc[0])
        return df
    
    def evolve(self):
        """Run the genetic algorithm"""
        print(f"Starting evolution with {self.population_size} chromosomes for {self.generations} generations")
        start_time = time.time()
        
        # Initialize population
        self._initialize_population()
        
        # Setup progress reporter
        progress = EvolutionProgressReporter(self.generations)
        
        # Evaluate initial population
        print("Evaluating initial population...")
        def evaluate_parallel(chrom):
            return self._evaluate_chromosome(chrom)

        self.fitness_scores = Parallel(n_jobs=-1)(
            delayed(evaluate_parallel)(chrom) for chrom in self.population
        )
        
        print(f"Initial fitness scores range: {min(self.fitness_scores):.6f} to {max(self.fitness_scores):.6f}")
        print(f"Average initial fitness: {np.mean(self.fitness_scores):.6f}")
        
        # Track the best chromosome
        best_idx = np.argmax(self.fitness_scores)
        self.best_chromosome = deepcopy(self.population[best_idx])
        self.best_fitness = self.fitness_scores[best_idx]
        
        print(f"Initial best fitness: {self.best_fitness:.6f}")
        
        # Add expert features that might be valuable
        expert_features = []

        # Create specific rainfall-related features known to be valuable
        if 'rainfall_lag1' in self.base_features and 'cloud' in self.base_features:
            expert_features.append(('multiply', 'rainfall_lag1', 'cloud'))

        if 'humidity' in self.base_features and 'temparature' in self.base_features:
            expert_features.append(('multiply', 'humidity', 'temparature'))

        # Add to best chromosome if it improves fitness
        if expert_features:
            test_chromosome = self.best_chromosome.copy()
            for feature in expert_features:
                test_chromosome.append(feature)
                
            test_fitness = self._evaluate_chromosome(test_chromosome)
            if test_fitness > self.best_fitness:
                self.best_chromosome = test_chromosome
                self.best_fitness = test_fitness
                print(f"Added expert features, new fitness: {self.best_fitness:.6f}")
        
        # Evolution loop
        for generation in range(self.generations):
            # Create new population
            new_population = []
            
            # Elitism: keep the best chromosome
            new_population.append(deepcopy(self.best_chromosome))
            
            # Create offspring
            while len(new_population) < self.population_size:
                # Select parents
                parent1, parent2 = self._select_parents()
                
                # Crossover
                child1, child2 = self._crossover(parent1, parent2)
                
                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                # Add to new population
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # Update population
            self.population = new_population
            
            # Evaluate new population
            self.fitness_scores = Parallel(n_jobs=-1)(
                delayed(evaluate_parallel)(chrom) for chrom in self.population
            )
            
            # Update best chromosome
            best_idx = np.argmax(self.fitness_scores)
            if self.fitness_scores[best_idx] > self.best_fitness:
                self.best_chromosome = deepcopy(self.population[best_idx])
                self.best_fitness = self.fitness_scores[best_idx]
            
            # Track history
            avg_fitness = np.mean(self.fitness_scores)
            self.fitness_history.append({
                'generation': generation + 1,
                'best_fitness': self.best_fitness,
                'avg_fitness': avg_fitness
            })
            
            # Generation summary
            self.generation_summaries.append({
                'generation': generation + 1,
                'best_fitness': self.best_fitness,
                'avg_fitness': avg_fitness,
                'best_chromosome_size': len(self.best_chromosome)
            })
            
            # Save progress
            if (generation + 1) % 5 == 0:
                self._save_progress()
                
            # Update progress
            progress.update(generation+1, self.best_fitness, len(self.best_chromosome), avg_fitness)
        
        # Close progress bar
        progress.close()
        
        # Final results
        elapsed_time = time.time() - start_time
        print(f"\nEvolution completed in {elapsed_time:.1f} seconds")
        print(f"Best fitness: {self.best_fitness:.6f}")
        print(f"Best chromosome has {len(self.best_chromosome)} features")
        
        # Save final results
        self._save_progress(final=True)
        
        return self.best_chromosome, self.best_fitness
    
    def _save_progress(self, final=False):
        """Save progress to disk"""
        # Save fitness history
        fitness_df = pd.DataFrame(self.fitness_history)
        fitness_df.to_csv('genetic_engineering/fitness_history.csv', index=False)
        
        # Save generation summaries
        summary_df = pd.DataFrame(self.generation_summaries)
        summary_df.to_csv('genetic_engineering/generation_summaries.csv', index=False)
        
        # Save best chromosome
        if final:
            with open('genetic_engineering/best_chromosome.pkl', 'wb') as f:
                pickle.dump({
                    'chromosome': self.best_chromosome,
                    'fitness': self.best_fitness,
                    'transformers': self.feature_transformers
                }, f)
            
            # Also save human-readable version
            feature_names = []
            for feature in self.best_chromosome:
                if isinstance(feature, str):
                    feature_names.append(feature)
                else:
                    op, *operands = feature
                    name = self._generate_feature_name(op, operands)
                    feature_names.append(name)
            
            with open('genetic_engineering/best_features.txt', 'w') as f:
                f.write(f"Best fitness: {self.best_fitness:.6f}\n\n")
                f.write("Features:\n")
                for i, name in enumerate(feature_names):
                    f.write(f"{i+1}. {name}\n")
    
    def apply_to_test_data(self):
        """Apply the best feature set to test data"""
        print("\nApplying best feature set to test data...")
        
        # Create feature matrices
        X_train_final = self.X_train.copy()
        X_test_final = self.X_test.copy()
        
        # Apply transformations from the best chromosome
        feature_set = []
        
        for feature in self.best_chromosome:
            if isinstance(feature, str):
                # Base feature
                if feature in self.base_features:
                    feature_set.append(feature)
            else:
                # Derived feature
                operation, operands = feature[0], feature[1:]
                feature_name = self._generate_feature_name(operation, operands)
                
                # Apply to train
                X_train_final[feature_name] = self._apply_transformation(
                    operation, operands, X_train_final)
                
                # Apply to test
                X_test_final[feature_name] = self._apply_transformation(
                    operation, operands, X_test_final)
                
                feature_set.append(feature_name)
        
        # Fill missing values
        X_train_final = self._fill_missing_values(X_train_final[feature_set])
        X_test_final = self._fill_missing_values(X_test_final[feature_set])
        
        print(f"Created test data with {len(feature_set)} features")
        
        return X_train_final, X_test_final, feature_set
    
    def create_submission(self, X_test_temporal):
        """Create a submission using the best features"""
        # Apply features to test data
        X_train_final, X_test_final, feature_set = self.apply_to_test_data()
        
        # Train models with different parameters
        print("Training multiple models for ensemble...")
        models = []
        
        # Model 1: RandomForest
        rf = RandomForestRegressor(
            n_estimators=1000, 
            max_depth=None, 
            min_samples_leaf=1,
            max_features=0.7,
            n_jobs=-1, 
            random_state=42
        )
        rf.fit(X_train_final, self.y_train)
        models.append(("RF", rf))
        
        # Model 2: LightGBM (if available)
        try:
            import lightgbm as lgb
            lgbm = lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=-1,
                num_leaves=31,
                n_jobs=-1,
                random_state=43
            )
            lgbm.fit(X_train_final, self.y_train)
            models.append(("LGBM", lgbm))
        except ImportError:
            print("LightGBM not available, skipping")
        
        # Make predictions with each model
        print("Making predictions...")
        preds = []
        for name, model in models:
            pred = model.predict(X_test_final)
            preds.append(pred)
        
        # Create an optimal ensemble
        final_preds = np.mean(preds, axis=0)
        
        # Create submission
        submission = pd.DataFrame({
            'id': X_test_temporal['id'],
            'rainfall': final_preds
        })
        
        # Load best existing model for blending (if available)
        try:
            best_model = pd.read_csv("/home/USER/Projects/RAIN/data/candidates/submission.85518.csv")
            best_preds = best_model['rainfall'].values
            
            # Create blended submission - this ratio derived from multiple experiments
            blend = pd.DataFrame({
                'id': X_test_temporal['id'],
                'rainfall': 0.7 * best_preds + 0.3 * final_preds
            })
            
            # Save blend
            blend.to_csv('/home/USER/Projects/RAIN/data/candidates/genetic_blend_features.csv', index=False)
            print("Created genetic blend submission")
        except Exception as e:
            print(f"Could not create blend: {e}")
        
        # Save pure genetic features submission
        submission.to_csv('/home/USER/Projects/RAIN/data/candidates/genetic_pure_features.csv', index=False)
        print("Created pure genetic features submission")
        
        return submission, X_train_final, X_test_final, feature_set

# Run the genetic algorithm
if __name__ == "__main__":
    from data_pipeline import create_complete_data_pipeline
    
    # Load data
    X_train_temporal, X_test_temporal = create_complete_data_pipeline()
    y_train = X_train_temporal['rainfall']
    X_train = X_train_temporal.drop(['id', 'rainfall'], axis=1)
    X_test = X_test_temporal.drop(['id'], axis=1)
    
    # Initialize and run GA
    ga = GeneticFeatureEngineering(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        population_size=400,
        generations=200,  # Reduce for faster execution
        mutation_rate=0.3,
        crossover_rate=0.7
    )
    
    # Run evolution
    best_chromosome, best_fitness = ga.evolve()
    
    # Create submission
    submission, X_train_final, X_test_final, feature_set = ga.create_submission(X_test_temporal)
    
    print(f"Final validation score: {best_fitness:.6f}")
    print("Genetic feature engineering complete!")