import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from genetic_feature_engineering import GeneticFeatureEngineering
from data_pipeline import create_complete_data_pipeline

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    X_train_temporal, X_test_temporal = create_complete_data_pipeline()
    y_train = X_train_temporal['rainfall']
    X_train = X_train_temporal.drop(['id', 'rainfall'], axis=1)
    X_test = X_test_temporal.drop(['id'], axis=1)
    
    # Initialize and run GA with different parameters
    print("Initializing GA...")
    ga = GeneticFeatureEngineering(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        population_size=100,  # Reduced for faster iterations
        generations=50,      # Reduced for faster completion
        mutation_rate=0.3,
        crossover_rate=0.7
    )
    
    # Run evolution
    best_chromosome, best_fitness = ga.evolve()
    
    # Create submission
    submission, X_train_final, X_test_final, feature_set = ga.create_submission(X_test_temporal)
    
    print(f"Final validation score: {best_fitness:.6f}")
    print("Top 20 features:")
    feature_names = []
    for feature in best_chromosome[:20]:
        if isinstance(feature, str):
            feature_names.append(feature)
        else:
            op, *operands = feature
            name = ga._generate_feature_name(op, operands)
            feature_names.append(name)
    
    for i, name in enumerate(feature_names):
        print(f"{i+1}. {name}")
    
    print("Genetic feature engineering complete!")