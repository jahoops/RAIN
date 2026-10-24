import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import time
from copy import deepcopy
import random

from simple_ga_validation import SimpleTrainValidator

class RobustGeneticFeatureEngineering:
    def __init__(self, X_train, y_train, X_test, 
                 population_size=30, generations=15, 
                 mutation_rate=0.3, crossover_rate=0.7):
        """Initialize the genetic algorithm with safeguards against overfitting"""
        # Create a proper train/validation split
        self.X_train_full = X_train.copy()
        self.y_train_full = y_train.copy()
        self.X_test = X_test.copy()
        
        # Create a proper train/validation split with stratification
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42
        )
        
        # Store base feature names
        self.base_features = list(X_train.columns)
        
        # GA parameters
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Initialize tracking variables
        self.population = []
        self.fitness_scores = []
        self.best_chromosome = None
        self.best_fitness = 0.0
        self.feature_transformers = {}
        
        # Initialize validator
        self.validator = SimpleTrainValidator(
            X_train=X_train,
            y_train=y_train
        )
        
        print(f"Initialized GA with {len(self.base_features)} base features")
        print(f"Train set: {self.X_train.shape}, Validation set: {self.X_val.shape}")
        
    def _generate_feature_name(self, operation, operands):
        """Generate a descriptive name for a feature"""
        if operation == 'log':
            return f'log_{operands[0]}'
        elif operation == 'sqrt':
            return f'sqrt_{operands[0]}'
        elif operation == 'square':
            return f'square_{operands[0]}'
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
        elif operation == 'rolling_mean':
            return f'{operands[0]}_roll{operands[1]}_mean'
        else:
            return f'{operation}_{"_".join(map(str, operands))}'

    def _apply_transformation(self, operation, operands, data):
        """Apply transformation with safeguards against data leakage"""
        try:
            # Check that we're not directly accessing the target
            for operand in operands:
                if isinstance(operand, str) and 'rainfall' in operand:
                    # Don't allow direct rainfall features to prevent leakage
                    return pd.Series(np.nan, index=data.index)
            
            if operation == 'log':
                # Log transform with safeguard for non-positive values
                x = data[operands[0]]
                min_val = x.min()
                if min_val <= 0:
                    offset = abs(min_val) + 1e-6
                    return np.log(x + offset)
                return np.log(x)
                
            elif operation == 'sqrt':
                # Square root with safeguard for negative values
                x = data[operands[0]]
                if (x < 0).any():
                    return np.sqrt(x - x.min() + 1e-6)
                return np.sqrt(x)
                
            elif operation == 'square':
                return data[operands[0]] ** 2
                
            elif operation == 'multiply':
                # Check for different columns to avoid perfect collinearity
                if operands[0] == operands[1]:
                    return data[operands[0]] ** 2
                return data[operands[0]] * data[operands[1]]
                
            elif operation == 'divide':
                # Division with safeguard for zeros
                denominator = data[operands[1]]
                if (denominator.abs() < 1e-10).any():
                    return data[operands[0]] / (denominator.abs() + 1e-10)
                return data[operands[0]] / denominator
                
            elif operation == 'add':
                return data[operands[0]] + data[operands[1]]
                
            elif operation == 'subtract':
                return data[operands[0]] - data[operands[1]]
                
            elif operation == 'lag':
                # Create a lag feature
                return data[operands[0]].shift(operands[1])
                
            elif operation == 'rolling_mean':
                # Rolling mean with minimum periods
                window = operands[1]
                return data[operands[0]].rolling(window=window, min_periods=1).mean()
                
            else:
                # Unrecognized operation
                return pd.Series(np.nan, index=data.index)
                
        except Exception as e:
            # Return NaN for any errors
            return pd.Series(np.nan, index=data.index)
    
    def _initialize_population(self):
        """Initialize the population with diverse chromosomes"""
        print("Initializing population with diverse chromosomes...")
        
        # Start with an empty population
        self.population = []
        
        # Create groups of chromosomes with varying feature counts
        min_features = 3
        max_features = min(15, len(self.base_features))
        
        for feature_count in range(min_features, max_features+1, 2):
            # Create chromosomes with this feature count
            for _ in range(self.population_size // ((max_features-min_features)//2 + 1)):
                # Select base features
                base_count = min(feature_count - 2, len(self.base_features))
                base_count = max(3, base_count)  # At least 3 features
                
                chromosome = random.sample(
                    self.base_features,
                    k=base_count
                )
                
                # Add derived features to reach target count
                derived_count = feature_count - len(chromosome)
                for _ in range(derived_count):
                    derived_feature = self._create_random_feature()
                    if derived_feature:
                        chromosome.append(derived_feature)
                
                self.population.append(chromosome)
        
        # Fill remaining slots
        while len(self.population) < self.population_size:
            chromosome = random.sample(
                self.base_features,
                k=random.randint(3, min(8, len(self.base_features)))
            )
            self.population.append(chromosome)
    
    def _create_random_feature(self):
        """Create a random derived feature"""
        # Select operation type
        operations = [
            # Unary operations
            ('log', 1), ('sqrt', 1), ('square', 1),
            # Binary operations
            ('multiply', 2), ('divide', 2), ('add', 2), ('subtract', 2),
            # Time-based operations
            ('lag', 2), ('rolling_mean', 2)
        ]
        
        try:
            # Select random operation
            operation, num_args = random.choice(operations)
            
            if num_args == 1:
                # Unary operation
                feature = random.choice(self.base_features)
                return (operation, feature)
            else:
                # Binary operation
                feature1 = random.choice(self.base_features)
                feature2 = random.choice(self.base_features)
                
                if operation == 'lag':
                    # For lag operations, use lag periods
                    lag = random.randint(1, 5)
                    return (operation, feature1, lag)
                elif operation == 'rolling_mean':
                    # For rolling operations, use window sizes
                    window = random.choice([3, 5, 7, 10])
                    return (operation, feature1, window)
                else:
                    # Normal binary operation
                    return (operation, feature1, feature2)
        except Exception:
            return None
    
    def _evaluate_chromosome(self, chromosome, generation, verbose=False):
        """Evaluate a chromosome with robust safeguards"""
        results = self.validator.validate_chromosome(
            chromosome=chromosome,
            generation=generation, 
            verbose=verbose
        )
        return results['fitness']
    
    def evolve(self):
        """Run the genetic algorithm with progress tracking"""
        print(f"Starting evolution with {self.population_size} chromosomes for {self.generations} generations")
        start_time = time.time()
        
        # Initialize population
        self._initialize_population()
        
        # Evaluate initial population
        print("Evaluating initial population...")
        self.fitness_scores = [self._evaluate_chromosome(chrom, 0) for chrom in self.population]
        
        # Track the best chromosome
        best_idx = np.argmax(self.fitness_scores)
        self.best_chromosome = deepcopy(self.population[best_idx])
        self.best_fitness = self.fitness_scores[best_idx]
        
        print(f"Initial best fitness: {self.best_fitness:.6f}")
        print(f"Initial fitness range: {min(self.fitness_scores):.6f} to {max(self.fitness_scores):.6f}")
        
        # Evolution loop
        for gen in range(self.generations):
            # Create new population
            new_population = [deepcopy(self.best_chromosome)]  # Elitism
            
            while len(new_population) < self.population_size:
                # Tournament selection
                tournament_size = 3
                parent1_idx = random.randint(0, len(self.population) - 1)
                for _ in range(tournament_size - 1):
                    idx = random.randint(0, len(self.population) - 1)
                    if self.fitness_scores[idx] > self.fitness_scores[parent1_idx]:
                        parent1_idx = idx
                
                parent2_idx = random.randint(0, len(self.population) - 1)
                for _ in range(tournament_size - 1):
                    idx = random.randint(0, len(self.population) - 1)
                    if self.fitness_scores[idx] > self.fitness_scores[parent2_idx]:
                        parent2_idx = idx
                
                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]
                
                # Crossover
                if random.random() < self.crossover_rate:
                    point = random.randint(1, min(len(parent1), len(parent2)) - 1)
                    child1 = parent1[:point] + parent2[point:]
                    child2 = parent2[:point] + parent1[point:]
                else:
                    child1, child2 = deepcopy(parent1), deepcopy(parent2)
                
                # Mutation
                if random.random() < self.mutation_rate:
                    if len(child1) > 3 and random.random() < 0.5:
                        # Remove a feature
                        idx = random.randint(0, len(child1) - 1)
                        child1.pop(idx)
                    else:
                        # Add a feature
                        if random.random() < 0.5:
                            # Add a base feature
                            available = [f for f in self.base_features if f not in child1]
                            if available:
                                child1.append(random.choice(available))
                        else:
                            # Add a derived feature
                            derived = self._create_random_feature()
                            if derived:
                                child1.append(derived)
                
                # Add children to new population
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # Update population
            self.population = new_population
            
            # Evaluate new population
            self.fitness_scores = [self._evaluate_chromosome(chrom, gen+1) for chrom in self.population]
            
            # Update best chromosome
            best_idx = np.argmax(self.fitness_scores)
            if self.fitness_scores[best_idx] > self.best_fitness:
                self.best_chromosome = deepcopy(self.population[best_idx])
                self.best_fitness = self.fitness_scores[best_idx]
            
            # Track generation
            self.validator.track_generation(
                generation=gen+1,
                population=self.population,
                fitness_scores=self.fitness_scores,
                best_chromosome=self.best_chromosome
            )
            
            # Report progress
            avg_fitness = np.mean(self.fitness_scores)
            print(f"Generation {gen+1}/{self.generations}: Best={self.best_fitness:.6f}, Avg={avg_fitness:.6f}, Features={len(self.best_chromosome)}")
        
        # Final validation with verbose output
        print("\nFinal detailed evaluation of best chromosome:")
        final_results = self.validator.final_validation(self.best_chromosome)
        
        elapsed = time.time() - start_time
        print(f"Evolution completed in {elapsed:.1f} seconds")
        
        return self.best_chromosome, self.best_fitness
    
    def create_submission(self, X_test_temporal):
        """Create a submission file with the best chromosome"""
        print("\nCreating submission with best chromosome...")
        
        # Create feature matrices
        X_train_full = self.X_train_full.copy()
        X_test = self.X_test.copy()
        
        # Apply transformations from the best chromosome
        feature_set = []
        
        # Process base features first
        for feature in self.best_chromosome:
            if isinstance(feature, str) and feature in self.base_features:
                feature_set.append(feature)
        
        # Process derived features
        for feature in self.best_chromosome:
            if not isinstance(feature, str):
                try:
                    operation, operands = feature[0], feature[1:]
                    feature_name = self._generate_feature_name(operation, operands)
                    
                    # Apply transformations
                    X_train_full[feature_name] = self._apply_transformation(operation, operands, X_train_full)
                    X_test[feature_name] = self._apply_transformation(operation, operands, X_test)
                    
                    # Check for valid features
                    if (X_train_full[feature_name].isna().sum() < len(X_train_full) * 0.5 and 
                        X_test[feature_name].isna().sum() < len(X_test) * 0.5):
                        feature_set.append(feature_name)
                except Exception as e:
                    print(f"Error applying feature {operation}: {e}")
        
        print(f"Using {len(feature_set)} features for final model")
        
        # Handle missing values
        for col in feature_set:
            if X_train_full[col].isna().sum() > 0:
                X_train_full[col] = X_train_full[col].fillna(X_train_full[col].median())
            if X_test[col].isna().sum() > 0:
                X_test[col] = X_test[col].fillna(X_train_full[col].median())
        
        # Train final model
        print("Training final model...")
        final_model = RandomForestRegressor(
            n_estimators=1000,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=87
        )
        
        final_model.fit(X_train_full[feature_set], self.y_train_full)
        
        # Make predictions
        predictions = final_model.predict(X_test[feature_set])
        
        # Create submission
        submission = pd.DataFrame({
            'id': X_test_temporal['id'],
            'rainfall': predictions
        })
        
        # Save the submission
        submission.to_csv('/home/USER/Projects/RAIN/data/candidates/robust_genetic_features.csv', index=False)
        print("Created submission with genetically engineered features")
        
        return submission

if __name__ == "__main__":
    # Import data pipeline only when needed
    from data_pipeline import create_complete_data_pipeline
    
    # Load data
    print("Loading data...")
    X_train_temporal, X_test_temporal = create_complete_data_pipeline()
    y_train = X_train_temporal['rainfall']
    X_train = X_train_temporal.drop(['id', 'rainfall'], axis=1)
    X_test = X_test_temporal.drop(['id'], axis=1)
    
    # Test a basic model for baseline
    print("Establishing baseline performance...")
    X_train_small, X_val_small, y_train_small, y_val_small = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train_small, y_train_small)
    val_pred = rf.predict(X_val_small)
    base_mse = mean_squared_error(y_val_small, val_pred)
    print(f"Baseline MSE: {base_mse:.6f}, Fitness: {1-base_mse:.6f}")
    
    # Initialize GA
    ga = RobustGeneticFeatureEngineering(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        population_size=400,
        generations=20,
        mutation_rate=0.3,
        crossover_rate=0.7
    )
    
    # Run GA
    best_chromosome, best_fitness = ga.evolve()
    
    # Create submission
    if best_fitness > 0.1:  # Only create submission if fitness is reasonable
        submission = ga.create_submission(X_test_temporal)
        
        # Print top features
        print("\nTop features in best chromosome:")
        feature_names = []
        for feature in best_chromosome[:10]:
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