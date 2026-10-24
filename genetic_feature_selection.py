import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os
import random
from tqdm import tqdm
from optimized_binary_model import create_binary_optimized_features
import matplotlib.pyplot as plt
import seaborn as sns
from ga_visualizations import visualize_ga_results

def island_genetic_selection():
    """Use island model genetic algorithm to find optimal feature subset"""
    # Load data
    train = pd.read_csv('/home/USER/Projects/RAIN/data/train.csv')
    test = pd.read_csv('/home/USER/Projects/RAIN/data/test.csv')

    def enhanced_FE(df):
        df = df.copy()
        
        # Existing feature
        df['cloudtest'] = (df.cloud==88).astype(int)
        
        # Cloud features
        df['overcast'] = (df.cloud >= 75).astype(int)
        
        # Pressure features
        df['low_pressure'] = (df.pressure < 1000).astype(int)
        
        # Humidity features
        df['high_humidity'] = (df.humidity > 85).astype(int)
        
        # Temperature-related
        if 'dewpoint' in df.columns:
            df['high_rain_potential'] = ((df.temparature - df.dewpoint) < 3).astype(int)
        
        # Wind-related
        df['strong_wind'] = (df.windspeed > 20).astype(int)
        
        # Combined factors
        df['rain_likely'] = ((df.humidity > 85) & (df.pressure < 1000)).astype(int)
        
        return df
    
    train = enhanced_FE(train)
    test = enhanced_FE(test)

    # Generate features
    X, X_test = create_binary_optimized_features(train, test)
    y = train['rainfall']

    # Apply SMOTE for class balance
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    
    # Create preprocessing pipeline
    preprocessing = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Base model
    lr = LogisticRegression(C=0.5, max_iter=10000, solver='liblinear', random_state=42)
    
    # Get all available features
    all_features = list(X.columns)
    
    # Get baseline performance
    X_processed = preprocessing.fit_transform(X)
    baseline_scores = cross_val_score(lr, X_processed, y, cv=kf, scoring='roc_auc')
    baseline_auc = np.mean(baseline_scores)
    print(f"Baseline AUC with all {len(all_features)} features: {baseline_auc:.6f}")
    
    # Calculate feature importance to guide mutation
    print("Calculating initial feature importance...")
    temp_model = LogisticRegression(C=0.5, max_iter=10000, solver='liblinear', random_state=42)
    temp_model.fit(X_processed, y)
    
    # Extract absolute feature importance
    importance_values = np.abs(temp_model.coef_[0])
    feature_importance = {feature: imp for feature, imp in zip(all_features, importance_values)}
    
    # Normalize to [0,1] range
    max_imp = max(feature_importance.values())
    feature_importance = {f: imp/max_imp for f, imp in feature_importance.items()}
    
    # Save initial feature importance
    init_importance = pd.DataFrame({
        'feature': list(feature_importance.keys()),
        'importance': list(feature_importance.values())
    }).sort_values('importance', ascending=False)
    init_importance.to_csv("data/candidates/initial_feature_importance.csv", index=False)
    
    # ----- START GENETIC ALGORITHM -----
    
    # Island Model GA Parameters
    NUM_ISLANDS = 5
    ISLAND_SIZE = 100  # 400 individuals per island = 2000 total
    ISOLATION_PERIOD = 10  # No migration for first 20 generations
    MIGRATION_PROBABILITY = 1/5000  # 1/20000 chance of migration after isolation
    PERIODIC_REFRESH_RATE = 0.1  # 10% of the population will be refreshed periodically
    
    # Standard GA Parameters
    MIN_GENES = 10
    GENERATIONS = 50
    TOURNAMENT_SIZE = 5
    MUTATION_RATE = 0.2
    CROSSOVER_RATE = 0.8
    ELITE_SIZE = 2  # Number of elites per island
    FEATURE_PENALTY = 0.0002  # Penalty per feature

    # Island-specific parameters
    MUTATION_RATES = [0.1, 0.15, 0.2, 0.25, 0.3]
    TOURNAMENT_SIZES = [5, 6, 7, 8, 9]
    DIVERSITY_BONUS = 0.01

    # Function to create a random chromosome with bias toward important features
    def create_chromosome(features, min_genes=10, feature_probs=None):
        chromosome = [0] * len(features)
        if feature_probs is None:
            probs = [feature_importance.get(f, 0.01) for f in features]
        else:
            probs = [feature_probs.get(f, 0.01) for f in features]
        probs = np.array(probs) / sum(probs)  # Normalize
        n_to_select = min_genes
        indices = np.random.choice(
            range(len(features)), 
            size=n_to_select, 
            replace=False, 
            p=probs
        )
        for idx in indices:
            chromosome[idx] = 1
        return chromosome
    
    # Function to evaluate fitness (AUC score) with penalty for feature count
    def evaluate_fitness(chromosome, features, X, y, preprocessing, model, cv):
        feature_count = sum(chromosome)
        selected_indices = [i for i, gene in enumerate(chromosome) if gene == 1]
        if feature_count < MIN_GENES:
            return 0
        selected_features = [features[i] for i in selected_indices]
        X_selected = X[selected_features]
        X_processed = preprocessing.fit_transform(X_selected)
        try:
            scores = cross_val_score(model, X_processed, y, cv=cv, scoring='roc_auc')
            auc = np.mean(scores)
            feature_penalty = FEATURE_PENALTY * feature_count
            return auc - feature_penalty
        except Exception as e:
            print(f"Error in fitness evaluation: {e}")
            return 0
    
    # Function for tournament selection
    def tournament_selection(population, fitnesses, tournament_size):
        selected = []
        for _ in range(len(population)):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
            selected.append(population[winner_idx].copy())
        return selected
    
    # Add a diversity-based selection component
    def diversity_aware_tournament_selection(population, fitnesses, tournament_size, island_idx):
        selected = []
        
        for _ in range(len(population)):
            # Standard tournament selection
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            
            # Calculate diversity score for each individual in tournament
            diversity_scores = []
            for idx in tournament_indices:
                individual = population[idx]
                # Calculate average Hamming distance to other individuals in population
                sample_size = min(20, len(population))
                others = random.sample(population, sample_size)
                distances = [sum(a != b for a, b in zip(individual, other)) for other in others]
                avg_distance = sum(distances) / len(distances) if distances else 0
                normalized_distance = avg_distance / len(individual)
                
                # Add diversity bonus to fitness
                adjusted_fitness = fitnesses[idx] + DIVERSITY_BONUS * normalized_distance
                diversity_scores.append(adjusted_fitness)
            
            # Select winner based on adjusted fitness
            winner_idx = tournament_indices[np.argmax(diversity_scores)]
            selected.append(population[winner_idx].copy())
        
        return selected

    # Function for crossover
    def crossover(parent1, parent2):
        if random.random() < CROSSOVER_RATE:
            crossover_point = random.randint(1, len(parent1) - 2)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            
            # Ensure minimum genes
            for child in [child1, child2]:
                if sum(child) < MIN_GENES:
                    available_zeros = [i for i, gene in enumerate(child) if gene == 0]
                    if available_zeros:
                        to_add = MIN_GENES - sum(child)
                        to_add = min(to_add, len(available_zeros))
                        weights = [feature_importance.get(all_features[i], 0.01) for i in available_zeros]
                        weights = np.array(weights) / sum(weights)
                        to_add_indices = np.random.choice(
                            available_zeros, 
                            size=to_add, 
                            replace=False,
                            p=weights
                        )
                        for idx in to_add_indices:
                            child[idx] = 1
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    # Update your mutation function to accept variable rates
    def mutate(chromosome, mutation_rate=None, homogeneity_factor=None):
        """Mutate a chromosome with a given mutation rate"""
        if mutation_rate is None:
            mutation_rate = MUTATION_RATE
        
        # Apply homogeneity factor if provided
        if homogeneity_factor is not None:
            # Increase mutation rate for homogeneous populations
            mutation_rate *= homogeneity_factor
        
        for i in range(len(chromosome)):
            base_prob = mutation_rate
            importance = feature_importance.get(all_features[i], 0.01)
            
            if chromosome[i] == 1:  # Feature is currently selected
                adjusted_prob = base_prob * (1 - 0.5 * importance)
            else:  # Feature is currently not selected
                adjusted_prob = base_prob * (1 + 0.5 * importance)
                
            if random.random() < adjusted_prob:
                chromosome[i] = 1 - chromosome[i]  # Flip
                
        # Ensure minimum genes after mutation
        if sum(chromosome) < MIN_GENES:
            available_zeros = [i for i, gene in enumerate(chromosome) if gene == 0]
            if available_zeros:
                to_add = MIN_GENES - sum(chromosome)
                to_add = min(to_add, len(available_zeros))
                weights = [feature_importance.get(all_features[i], 0.01) for i in available_zeros]
                weights = np.array(weights) / sum(weights)
                to_add_indices = np.random.choice(
                    available_zeros, 
                    size=to_add, 
                    replace=False,
                    p=weights
                )
                for idx in to_add_indices:
                    chromosome[idx] = 1
        return chromosome
    
    # Initialize islands with different biases
    islands = []
    for i in range(NUM_ISLANDS):
        # Each island starts with different feature preferences
        feature_subset = random.sample(all_features, len(all_features)//2)
        
        # Boost importance of these features for this island's initialization
        island_importance = feature_importance.copy()
        for f in feature_subset:
            if f in island_importance:
                island_importance[f] *= 2.0
        
        # Create island with these biases
        island = []
        for _ in range(ISLAND_SIZE):
            chrom = create_chromosome(all_features, MIN_GENES, feature_probs=island_importance)
            island.append(chrom)
            
        islands.append(island)
        print(f"Island {i+1} initialized with bias toward {len(feature_subset)} features")
    
    # Track best solution across all islands
    global_best_chromosome = None
    global_best_fitness = 0
    global_best_generation = 0
    global_best_island = 0
    
    # Track history for each island
    island_histories = [[] for _ in range(NUM_ISLANDS)]
    
    # Track feature prevalence over time
    feature_tracking = []
    homogeneity_tracking = []
    
    print(f"Running for {GENERATIONS} generations with {ISOLATION_PERIOD} isolation period...")
    
    for generation in tqdm(range(GENERATIONS)):
        # Calculate homogeneity for each island every generation
        island_homogeneity = []
        for island in islands:
            # Calculate average hamming distance between all pairs
            total_distance = 0
            pairs = 0
            # Sample 20 random individuals for efficiency
            sample_size = min(20, len(island))
            sample = random.sample(island, sample_size)
            
            for i in range(len(sample)):
                for j in range(i+1, len(sample)):
                    # Hamming distance between two chromosomes
                    distance = sum(a != b for a, b in zip(sample[i], sample[j]))
                    total_distance += distance
                    pairs += 1
            
            # Average distance (normalized by chromosome length)
            avg_distance = total_distance / (pairs * len(all_features)) if pairs > 0 else 0
            # Convert to homogeneity (1 = completely homogeneous, 0 = completely diverse)
            homogeneity = 1 - avg_distance
            island_homogeneity.append(homogeneity)
        
        # Track homogeneity over time (now moved before the island evolution loop)
        for i, h in enumerate(island_homogeneity):
            homogeneity_tracking.append({
                'generation': generation,
                'island': i,
                'homogeneity': h
            })
            
        # Print homogeneity stats every 10 generations
        if generation % 10 == 0:
            homogeneity_str = " | ".join([f"Island {i+1}: {h:.2f}" for i, h in enumerate(island_homogeneity)])
            print(f"Generation {generation} - Population homogeneity: {homogeneity_str}")
            
        # For each island
        for island_idx, island in enumerate(islands):
            # Use island-specific mutation rate and tournament size
            current_mutation_rate = MUTATION_RATES[island_idx]
            current_tournament_size = TOURNAMENT_SIZES[island_idx]
            
            # Evaluate fitness for each chromosome in this island
            fitnesses = [evaluate_fitness(chrom, all_features, X, y, preprocessing, lr, kf) 
                        for chrom in island]
            
            # Record this generation's statistics for this island
            max_fitness = max(fitnesses)
            avg_fitness = np.mean(fitnesses)
            best_idx = np.argmax(fitnesses)
            
            island_histories[island_idx].append({
                'generation': generation,
                'island': island_idx,
                'max_fitness': max_fitness,
                'avg_fitness': avg_fitness,
                'features_used': sum(island[best_idx])
            })
            
            # Update global best solution if improved
            if max_fitness > global_best_fitness:
                global_best_fitness = max_fitness
                global_best_chromosome = island[best_idx].copy()
                global_best_generation = generation
                global_best_island = island_idx
                print(f"New best solution: AUC = {max_fitness:.6f}, Features = {sum(global_best_chromosome)}, "
                     f"Island = {global_best_island+1}")
            
            # Elite selection for this island
            elite = []
            if ELITE_SIZE > 0:
                elite_indices = np.argsort(fitnesses)[-ELITE_SIZE:]
                elite = [island[i].copy() for i in elite_indices]
                
            # Selection
            selected = diversity_aware_tournament_selection(island, fitnesses, current_tournament_size, island_idx)
            
            # Create new island population
            new_island = []
            
            # Add elites first
            new_island.extend(elite)
            
            # Crossover and mutation for the rest
            for i in range(0, len(selected) - 1, 2):
                if i + 1 < len(selected):
                    child1, child2 = crossover(selected[i], selected[i + 1])
                    child1 = mutate(child1, mutation_rate=current_mutation_rate)
                    child2 = mutate(child2, mutation_rate=current_mutation_rate)
                    new_island.extend([child1, child2])
                    
            # If odd number, add the last one with mutation
            if len(selected) % 2 == 1:
                child = mutate(selected[-1], mutation_rate=current_mutation_rate)
                new_island.append(child)
                
            # Trim to island size if needed
            islands[island_idx] = new_island[:ISLAND_SIZE]
        
        # Migration between islands after isolation period
        if generation >= ISOLATION_PERIOD:
            # For each island, adjust migration rates based on homogeneity
            for source_island_idx in range(NUM_ISLANDS):
                # Base migration probability
                base_prob = MIGRATION_PROBABILITY
                
                # Adjust emigration probability based on homogeneity
                # More homogeneous populations are more likely to have emigrants
                source_homogeneity = island_homogeneity[source_island_idx]
                # Scale emigration rate (up to 10x for completely homogeneous population)
                emigration_factor = 1 + 9 * source_homogeneity
                emigration_prob = base_prob * emigration_factor
                
                migrants = []
                migrant_indices = []
                
                # Check each individual for potential migration using adjusted probability
                for i, individual in enumerate(islands[source_island_idx]):
                    if random.random() < emigration_prob:
                        migrants.append(individual)
                        migrant_indices.append(i)
                
                # If there are migrants
                if migrants:
                    # Replace the migrants in source island with new random individuals
                    for idx in migrant_indices:
                        islands[source_island_idx][idx] = create_chromosome(all_features, MIN_GENES)
                    
                    # For each migrant, find destination with weighted probability based on homogeneity
                    for migrant in migrants:
                        # Compute immigration weights for all islands except source
                        immigration_weights = []
                        possible_destinations = []
                        
                        for dest_idx in range(NUM_ISLANDS):
                            if dest_idx != source_island_idx:
                                possible_destinations.append(dest_idx)
                                # More homogeneous islands are more likely to accept immigrants
                                dest_homogeneity = island_homogeneity[dest_idx]
                                # Scale immigration rate (up to 5x for completely homogeneous population)
                                immigration_factor = 1 + 4 * dest_homogeneity
                                immigration_weights.append(immigration_factor)
                        
                        # Normalize weights
                        total_weight = sum(immigration_weights)
                        immigration_weights = [w/total_weight for w in immigration_weights] if total_weight > 0 else None
                        
                        # Choose destination based on weighted probability
                        dest_island_idx = np.random.choice(possible_destinations, p=immigration_weights)
                        
                        # Replace a random individual in the destination island
                        random_idx = random.randint(0, ISLAND_SIZE - 1)
                        islands[dest_island_idx][random_idx] = migrant.copy()
                    
                    print(f"Gen {generation}: {len(migrants)} individuals migrated from Island {source_island_idx+1} "
                          f"(homogeneity: {source_homogeneity:.2f}, emigration factor: {emigration_factor:.1f}x)")
        
        # Add this after the main evolution step for each island
        if generation % 10 == 0 and generation > 0:
            # Periodically refresh part of population with new random individuals
            for island_idx, island in enumerate(islands):
                num_to_refresh = int(ISLAND_SIZE * PERIODIC_REFRESH_RATE)
                if num_to_refresh > 0:
                    # Replace the worst-performing individuals
                    worst_indices = np.argsort(fitnesses)[:num_to_refresh]
                    for idx in worst_indices:
                        islands[island_idx][idx] = create_chromosome(all_features, MIN_GENES)
                    print(f"Gen {generation}: Refreshed {num_to_refresh} individuals in Island {island_idx+1}")
        
        # Track homogeneity over time
        for i, h in enumerate(island_homogeneity):
            homogeneity_tracking.append({
                'generation': generation,
                'island': i,
                'homogeneity': h
            })
        
        # Track feature prevalence
        feature_prevalence = {}
        for i, island in enumerate(islands):
            feature_prevalence['generation'] = generation
            feature_prevalence['island'] = i
            
            # Calculate prevalence of each feature
            for j, feature in enumerate(all_features):
                # Count chromosomes that have this feature
                feature_count = sum(chrom[j] == 1 for chrom in island)
                # Calculate percentage
                prevalence = feature_count / len(island)
                feature_prevalence[feature] = prevalence
            
            feature_tracking.append(feature_prevalence.copy())
        
        # Print progress every few generations
        if (generation + 1) % 5 == 0 or generation == 0:
            island_stats = []
            for i, history in enumerate(island_histories):
                if history:  # If there's history for this island
                    latest = history[-1]
                    island_stats.append(f"Island {i+1}: AUC={latest['max_fitness']:.6f}, Feats={latest['features_used']}")
            
            print(f"Generation {generation + 1}: Best global AUC = {global_best_fitness:.6f}")
            print(" | ".join(island_stats))
    
    # Evaluate final best solution without feature penalty
    print("\n=== ISLAND MODEL RESULTS ===")
    
    # Get selected features from best chromosome
    selected_indices = [i for i, gene in enumerate(global_best_chromosome) if gene == 1]
    selected_features = [all_features[i] for i in selected_indices]
    
    # Calculate raw AUC for final solution (without feature penalty)
    X_selected = X[selected_features]
    X_processed = preprocessing.fit_transform(X_selected)
    final_scores = cross_val_score(lr, X_processed, y, cv=kf, scoring='roc_auc')
    raw_auc = np.mean(final_scores)
    
    print(f"Best solution found in generation {global_best_generation + 1} on Island {global_best_island + 1}")
    print(f"Best AUC (with feature penalty): {global_best_fitness:.6f}")
    print(f"Best AUC (raw): {raw_auc:.6f}")
    print(f"Selected {len(selected_features)} out of {len(all_features)} features")
    print(f"Improvement over baseline: {raw_auc - baseline_auc:.6f}")
    
    # Save evolution history from all islands
    all_history = []
    for island_idx, history in enumerate(island_histories):
        all_history.extend(history)
    history_df = pd.DataFrame(all_history)
    history_df.to_csv("data/candidates/island_model_history.csv", index=False)
    
    # Save selected features
    pd.DataFrame({'feature': selected_features}).to_csv(
        "data/candidates/island_optimal_features.csv", index=False)
    
    # Train final model with selected features
    X_selected = X[selected_features]
    X_test_selected = X_test[selected_features]
    
    X_selected_processed = preprocessing.fit_transform(X_selected)
    X_test_selected_processed = preprocessing.transform(X_test_selected)
    
    # Train the final model
    print("\nTraining final model with selected features...")
    final_model = LogisticRegression(C=0.5, max_iter=10000, solver='liblinear', random_state=42)
    final_model.fit(X_selected_processed, y)
    
    # Generate predictions
    test_preds = final_model.predict_proba(X_test_selected_processed)[:,1]
    
    # Create submission
    submission = pd.DataFrame({
        'id': test['id'],
        'rainfall': test_preds
    })
    
    output_path = f"data/candidates/island_genetic_auc_{raw_auc:.6f}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    # Show selected features
    print("\nSelected features:")
    for feature in selected_features:
        print(f"- {feature}")
    
    # Display feature importance for selected features
    if hasattr(final_model, 'coef_'):
        importance = pd.DataFrame({
            'feature': selected_features,
            'coefficient': final_model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print("\nFeature importance:")
        print(importance.to_string(index=False))
        
        # Save feature importance
        importance.to_csv("data/candidates/island_feature_importance.csv", index=False)
    
    # Generate visualizations
    visualize_ga_results(
        island_histories=island_histories,
        feature_tracking=feature_tracking,
        homogeneity_tracking=homogeneity_tracking,
        all_features=all_features,
        selected_features=selected_features,
        output_dir="data/visualizations"
    )
    
    return selected_features, raw_auc

if __name__ == "__main__":
    island_genetic_selection()