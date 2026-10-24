import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

class SimpleTrainValidator:
    """Validator that alternates between real and synthetic data for evaluation"""
    
    def __init__(self, X_train, y_train):
        """Initialize with training data and generate synthetic data"""
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.generation_scores = []
        self.feature_history = {}
        self.synthetic_data_used = {}
        
        # Create directory for plots
        os.makedirs("analysis/ga_validation", exist_ok=True)
        print(f"Initialized validator with {X_train.shape[1]} base features")
        
        # Generate synthetic data
        self._generate_synthetic_data()
        
    def _generate_synthetic_data(self):
        """Generate truly challenging synthetic data for robust validation"""
        print("Generating challenging synthetic data...")
        
        # Make a copy of the original data
        X_synth = self.X_train.copy()
        
        # Step 1: Add significant noise (50% instead of 20%)
        for col in X_synth.columns:
            if X_synth[col].dtype.kind in 'fc':
                col_std = X_synth[col].std()
                noise = np.random.normal(0, 0.5 * col_std, size=len(X_synth))
                X_synth[col] = X_synth[col] + noise
        
        # Step 2: Reverse the importance of some features
        # Select 30% of features to invert their relationship
        feature_subset = np.random.choice(
            [c for c in X_synth.columns if X_synth[c].dtype.kind in 'fc'], 
            size=max(3, int(0.3 * len(X_synth.columns))), 
            replace=False
        )
        
        for col in feature_subset:
            # Invert the feature values (high becomes low, low becomes high)
            X_synth[col] = X_synth[col].max() + X_synth[col].min() - X_synth[col]
        
        print(f"Inverted relationships for features: {', '.join(feature_subset)}")
        
        # Step 3: Create synthetic target with VERY different feature relationships
        # Use mostly different features than what's likely important in the real data
        
        # Get proxy for feature importance in original data
        from sklearn.ensemble import RandomForestRegressor
        try:
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X_synth, self.y_train)
            importances = rf.feature_importances_
            
            # Get indices sorted by importance
            indices = np.argsort(importances)
            
            # Select features from BOTTOM 70% of importance
            bottom_features = [X_synth.columns[i] for i in indices[:int(0.7*len(indices))]]
            
            # Use these less-important features as the key drivers in synthetic data
            synth_features = np.random.choice(
                bottom_features,
                size=min(5, len(bottom_features)),
                replace=False
            )
        except:
            # Fallback if the above fails
            synth_features = np.random.choice(X_synth.columns, 5, replace=False)
        
        # Generate synthetic target using these different features
        y_synth = np.zeros(len(X_synth))
        
        for i, feature in enumerate(synth_features):
            # Use different weights and non-linear transformations
            weight = (i + 1) / len(synth_features)
            
            # Normalize the feature
            normalized = (X_synth[feature] - X_synth[feature].mean()) / max(X_synth[feature].std(), 1e-8)
            
            # Apply non-linear transformation
            if i % 3 == 0:
                transformed = np.square(normalized)  # Square
            elif i % 3 == 1:
                transformed = np.exp(np.clip(normalized, -3, 3) / 2)  # Exp transform
            else:
                transformed = np.sin(normalized * np.pi)  # Sine transform
            
            # Add to synthetic target with weight
            y_synth += weight * transformed
        
        # Transform to match rainfall distribution characteristics
        y_synth = np.exp(y_synth / 3)
        
        # Scale to match original range
        y_min, y_max = self.y_train.min(), self.y_train.max()
        y_synth = (y_synth - y_synth.min()) / (y_synth.max() - y_synth.min() + 1e-8)
        y_synth = y_synth * (y_max - y_min) + y_min
        
        # Force same percentage of zeros but at different positions
        zero_pct = (self.y_train == 0).mean()
        
        # Instead of setting lowest values to zero, randomly select indices
        np.random.seed(42)  # For reproducibility
        zero_indices = np.random.choice(
            range(len(y_synth)), 
            size=int(zero_pct * len(y_synth)),
            replace=False
        )
        y_synth[zero_indices] = 0
        
        # Store synthetic data
        self.X_synth = X_synth
        self.y_synth = pd.Series(y_synth, index=self.y_train.index)
        
        print(f"Created synthetic data using features: {', '.join(synth_features)}")
        print(f"Original y stats - mean: {self.y_train.mean():.4f}, std: {self.y_train.std():.4f}, zeros: {zero_pct*100:.1f}%")
        print(f"Synthetic y stats - mean: {self.y_synth.mean():.4f}, std: {self.y_synth.std():.4f}, zeros: {(self.y_synth == 0).mean()*100:.1f}%")
        
    def validate_chromosome(self, chromosome, generation=0, verbose=False):
        """Validation that alternates between real and synthetic data"""
        # Extract feature list
        features = self._extract_features(chromosome)
        
        if len(features) < 3:
            if verbose:
                print("Too few features to evaluate")
            return {'fitness': 0}
        
        # Determine if we should use synthetic data (every other generation)
        use_synthetic = (generation % 2 == 0) and generation > 0
        
        # Record which data was used
        self.synthetic_data_used[generation] = use_synthetic
        
        try:
            # Select appropriate datasets
            if use_synthetic:
                X_data = self.X_synth[features].copy()
                y_data = self.y_synth.copy()
                data_type = "synthetic"
            else:
                X_data = self.X_train[features].copy()
                y_data = self.y_train.copy()
                data_type = "real"
            
            # Fill missing values
            for col in X_data.columns:
                missing_count = X_data[col].isna().sum()
                if missing_count > 0:
                    X_data[col] = X_data[col].fillna(X_data[col].median())
            
            # Simple k-fold validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_scores = []
            
            for train_idx, val_idx in kf.split(X_data):
                # Split data
                X_train_fold = X_data.iloc[train_idx]
                y_train_fold = y_data.iloc[train_idx]
                X_val_fold = X_data.iloc[val_idx]
                y_val_fold = y_data.iloc[val_idx]
                
                # Train model - simple RandomForest
                model = RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=10,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(X_train_fold, y_train_fold)
                
                # Predict
                y_pred = model.predict(X_val_fold)
                
                # Calculate MSE
                mse = mean_squared_error(y_val_fold, y_pred)
                fold_scores.append(mse)
            
            # Use average MSE
            avg_mse = np.mean(fold_scores)
            
            # Simple fitness conversion (higher is better)
            raw_fitness = 1 - avg_mse
            
            # Cap at 0.95 (or gradually increase cap if using generation-based cap)
            fitness = min(raw_fitness, 0.95)
            
            if verbose:
                print(f"Features: {len(features)} - Data: {data_type} - MSE: {avg_mse:.6f} - Fitness: {fitness:.6f}")
                print(f"Feature set: {features}")
            
            # Track features for this generation
            self._update_feature_history(features, generation)
            
            return {
                'fitness': fitness, 
                'mse': avg_mse, 
                'features': features,
                'data_type': data_type
            }
            
        except Exception as e:
            if verbose:
                print(f"Error in validation: {e}")
            return {'fitness': 0}
    
    # The rest of the methods remain the same
    def _extract_features(self, chromosome):
        """Extract feature names from chromosome"""
        features = []
        
        for feature in chromosome:
            if isinstance(feature, str) and feature in self.X_train.columns:
                features.append(feature)
                
        return features
    
    def _update_feature_history(self, features, generation):
        """Track feature usage over generations"""
        if generation in self.feature_history:
            self.feature_history[generation].extend(features)
            self.feature_history[generation] = list(set(self.feature_history[generation]))
        else:
            self.feature_history[generation] = list(set(features))
    
    def track_generation(self, generation, population, fitness_scores, best_chromosome):
        """Track generation statistics"""
        # Extract features from best chromosome
        if best_chromosome:
            best_features = self._extract_features(best_chromosome)
            self._update_feature_history(best_features, generation)
        
        # Store the generation scores
        self.generation_scores.append({
            'generation': generation,
            'avg_fitness': np.mean(fitness_scores) if fitness_scores else 0,
            'max_fitness': np.max(fitness_scores) if fitness_scores else 0,
            'data_type': 'synthetic' if self.synthetic_data_used.get(generation, False) else 'real'
        })
        
        # Print generation stats
        top_fitness = sorted(fitness_scores, reverse=True)[:5]
        data_type = 'synthetic' if self.synthetic_data_used.get(generation, False) else 'real'
        print(f"Generation {generation} stats ({data_type} data):")
        print(f"  Top 5 fitness: {', '.join([f'{f:.4f}' for f in top_fitness])}")
        print(f"  Average fitness: {np.mean(fitness_scores):.4f}")
        
        # Create visualization every 5 generations
        if generation % 5 == 0 or generation == 1:
            self._plot_progress()
    
    def _plot_progress(self):
        """Plot GA progress with data type indicators"""
        if not self.generation_scores:
            return
        
        # Extract data
        generations = [g['generation'] for g in self.generation_scores]
        avg_fitness = [g['avg_fitness'] for g in self.generation_scores]
        max_fitness = [g['max_fitness'] for g in self.generation_scores]
        data_types = [g['data_type'] for g in self.generation_scores]
        
        # Create plot
        plt.figure(figsize=(12, 7))
        
        # Plot points with different markers based on data type
        for i, gen in enumerate(generations):
            marker = 'o' if data_types[i] == 'real' else 's'
            color = 'blue' if data_types[i] == 'real' else 'green'
            plt.plot(gen, avg_fitness[i], marker=marker, color=color, markersize=8)
            plt.plot(gen, max_fitness[i], marker=marker, color='red', markersize=8)
        
        # Connect the points
        plt.plot(generations, avg_fitness, 'b-', label='Average Fitness')
        plt.plot(generations, max_fitness, 'r-', label='Max Fitness')
        
        # Add legend for data types
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='gray', label='Real Data', markersize=8),
            Line2D([0], [0], marker='s', color='gray', label='Synthetic Data', markersize=8),
            Line2D([0], [0], color='red', label='Max Fitness'),
            Line2D([0], [0], color='blue', label='Avg Fitness')
        ]
        plt.legend(handles=legend_elements)
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('GA Fitness Progress with Alternating Data Types')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('analysis/ga_validation/ga_progress.png')
        plt.close()
    
    def final_validation(self, best_chromosome):
        """Final validation of best chromosome on BOTH datasets"""
        print("\n=== FINAL VALIDATION ===")
        
        # Get detailed validation results on real data
        real_results = self.validate_chromosome(best_chromosome, generation=-1, verbose=True)
        
        # Force evaluation on synthetic data
        self.synthetic_data_used[-1] = True
        synth_results = self.validate_chromosome(best_chromosome, generation=-1, verbose=True)
        
        # Reset to real data for the final reporting
        self.synthetic_data_used[-1] = False
        
        # Calculate robustness score - how consistent is performance across domains
        if 'fitness' in real_results and 'fitness' in synth_results:
            fitness_diff = abs(real_results['fitness'] - synth_results['fitness'])
            robustness = 1 - min(fitness_diff, 1)
            print(f"\nRobustness score: {robustness:.4f} (higher is better)")
            
            # Combine real and synthetic fitness with emphasis on real data
            combined_fitness = 0.7 * real_results['fitness'] + 0.3 * synth_results['fitness']
            print(f"Combined fitness: {combined_fitness:.4f}")
            
            real_results['robustness'] = robustness
            real_results['combined_fitness'] = combined_fitness
        
        # If we have a proper feature set, analyze importance
        features = self._extract_features(best_chromosome)
        if features and len(features) >= 3:
            try:
                X_features = self.X_train[features].copy()
                for col in X_features.columns:
                    if X_features[col].isna().sum() > 0:
                        X_features[col] = X_features[col].fillna(X_features[col].median())
                
                # Train a model on all data
                model = RandomForestRegressor(
                    n_estimators=200, 
                    max_depth=None,
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(X_features, self.y_train)
                
                # Print and plot feature importances
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                print("\nFeature ranking:")
                for i in range(len(features)):
                    print(f"{i+1}. {features[indices[i]]} ({importances[indices[i]]:.4f})")
                
                # Plot feature importances
                plt.figure(figsize=(12, 8))
                plt.title('Feature Importances')
                plt.bar(range(len(indices)), importances[indices], align='center')
                plt.xticks(range(len(indices)), [features[i] for i in indices], rotation=90)
                plt.tight_layout()
                plt.savefig('analysis/ga_validation/feature_importance.png')
                plt.close()
                
            except Exception as e:
                print(f"Error analyzing feature importance: {e}")
        
        return real_results