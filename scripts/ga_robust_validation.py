import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error, make_scorer
import seaborn as sns
import time
import os

class RobustGAValidator:
    """Robust validation framework for genetic algorithm feature engineering"""
    
    def __init__(self, X_train, y_train, X_test=None, temporal_column=None):
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.X_test = X_test.copy() if X_test is not None else None
        self.temporal_column = temporal_column
        
        # Create directories for analysis
        os.makedirs('analysis/ga_validation', exist_ok=True)
        
        # Track feature history
        self.feature_history = {}
        self.generation_scores = []
        
        print(f"Initialized validator with {X_train.shape[1]} base features")

    def validate_chromosome(self, chromosome, generation=0, verbose=False):
        """Multi-faceted validation of a chromosome"""
        
        # Extract feature list from chromosome
        features = self._extract_features(chromosome)
        
        if len(features) < 3:
            if verbose:
                print("Too few features to evaluate")
            return {
                'fitness': 0,
                'mse_mean': 1.0,
                'mse_std': 0.5,
                'complexity_penalty': 0,
                'temporal_score': 0,
                'diversity_score': 0,
                'physical_score': 0
            }
        
        # Apply features to create feature matrix
        X_features = self._apply_features(features)
        
        if X_features is None:
            if verbose:
                print("Failed to create feature matrix")
            return {
                'fitness': 0,
                'mse_mean': 1.0,
                'mse_std': 0.5,
                'complexity_penalty': 0,
                'temporal_score': 0,
                'diversity_score': 0,
                'physical_score': 0
            }
            
        # Track feature usage
        self._update_feature_history(features, generation)
            
        # Multiple validation components
        scores = {}
        
        # 1. Cross-validation for generalization
        cv_scores = self._cross_validation_score(X_features, features)
        scores.update(cv_scores)
        
        # 2. Temporal validation if temporal column available
        if self.temporal_column and self.temporal_column in self.X_train.columns:
            temporal_scores = self._temporal_validation(X_features, features)
            scores.update(temporal_scores)
        else:
            scores['temporal_score'] = 0.5  # Neutral if not applicable
        
        # 3. Feature complexity penalty
        scores['complexity_penalty'] = self._complexity_penalty(features)
        
        # 4. Feature diversity score
        scores['diversity_score'] = self._feature_diversity_score(features)
        
        # 5. Physical/meteorological validation
        scores['physical_score'] = self._physical_validation(X_features, features)
        
        # Calculate overall fitness with weighted components
        scores['fitness'] = self._calculate_fitness(scores)
        
        # Hard cap on fitness to prevent deception
        if scores['fitness'] > 0.95:
            scores['fitness'] = 0.95
            
        if verbose:
            self._print_validation_report(scores, features)
            
        return scores

    def _extract_features(self, chromosome):
        """Extract feature names from chromosome"""
        features = []
        
        for feature in chromosome:
            if isinstance(feature, str):
                # Base feature
                if feature in self.X_train.columns:
                    features.append(feature)
            elif isinstance(feature, tuple):
                # Derived feature - get name from operation
                try:
                    operation, operands = feature[0], feature[1:]
                    feature_name = self._generate_feature_name(operation, operands)
                    features.append(feature_name)
                except:
                    continue
                    
        return features
    
    def _generate_feature_name(self, operation, operands):
        """Generate feature name from operation"""
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
            return f'{operation}_{str(operands)}'
        
    def _apply_features(self, features):
        """Apply features to create feature matrix"""
        try:
            # Create copy of data with only selected features
            common_features = [f for f in features if f in self.X_train.columns]
            
            if len(common_features) < 3:
                return None
                
            X_features = self.X_train[common_features].copy()
            
            # Fill missing values
            for col in X_features.columns:
                if X_features[col].isna().sum() > 0:
                    if np.issubdtype(X_features[col].dtype, np.number):
                        X_features[col] = X_features[col].fillna(X_features[col].median())
                    else:
                        X_features[col] = X_features[col].fillna(X_features[col].mode().iloc[0])
                
            return X_features
        except Exception as e:
            print(f"Error applying features: {e}")
            return None
    
    def _cross_validation_score(self, X_features, features, n_splits=5):
        """Calculate cross-validation score with multiple folds"""
        try:
            # Use KFold cross-validation
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            mse_scores = []
            
            for train_idx, val_idx in kf.split(X_features):
                # Split data
                X_train_fold = X_features.iloc[train_idx]
                y_train_fold = self.y_train.iloc[train_idx]
                X_val_fold = X_features.iloc[val_idx]
                y_val_fold = self.y_train.iloc[val_idx]
                
                # Train model
                model = RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=10,
                    n_jobs=-1,
                    random_state=42
                )
                
                model.fit(X_train_fold, y_train_fold)
                
                # Predict
                y_pred = model.predict(X_val_fold)
                
                # Calculate MSE
                mse = mean_squared_error(y_val_fold, y_pred)
                mse_scores.append(mse)
            
            # Calculate statistics
            mse_mean = np.mean(mse_scores)
            mse_std = np.std(mse_scores)
            
            # Penalize high variance as it indicates instability
            stability = 1 - min(mse_std * 2, 0.5)  # Cap penalty at 0.5
            
            # CV score: higher is better
            cv_score = (1 - mse_mean) * stability
            
            return {
                'mse_mean': mse_mean,
                'mse_std': mse_std,
                'cv_score': cv_score
            }
            
        except Exception as e:
            print(f"Error in cross-validation: {e}")
            return {
                'mse_mean': 1.0,
                'mse_std': 0.5,
                'cv_score': 0
            }
    
    def _temporal_validation(self, X_features, features):
        """Validate with temporal splits to ensure time generalization"""
        try:
            if self.temporal_column not in X_features.columns:
                # Add temporal column if needed
                X_with_time = X_features.copy()
                X_with_time[self.temporal_column] = self.X_train[self.temporal_column].values
            else:
                X_with_time = X_features
                
            # Sort by temporal column
            sorted_indices = X_with_time[self.temporal_column].sort_values().index
            X_sorted = X_with_time.loc[sorted_indices].drop(columns=[self.temporal_column])
            y_sorted = self.y_train.loc[sorted_indices]
            
            # Create temporal CV splits
            tscv = TimeSeriesSplit(n_splits=3)
            
            mse_scores = []
            
            for train_idx, val_idx in tscv.split(X_sorted):
                # Split data
                X_train_fold = X_sorted.iloc[train_idx]
                y_train_fold = y_sorted.iloc[train_idx]
                X_val_fold = X_sorted.iloc[val_idx]
                y_val_fold = y_sorted.iloc[val_idx]
                
                # Train model
                model = RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=10,
                    n_jobs=-1,
                    random_state=42
                )
                
                model.fit(X_train_fold, y_train_fold)
                
                # Predict
                y_pred = model.predict(X_val_fold)
                
                # Calculate MSE
                mse = mean_squared_error(y_val_fold, y_pred)
                mse_scores.append(mse)
            
            # Calculate temporal score - emphasize last fold
            # (most recent data is most important)
            if len(mse_scores) >= 3:
                weighted_mse = 0.2 * mse_scores[0] + 0.3 * mse_scores[1] + 0.5 * mse_scores[-1]
            else:
                weighted_mse = np.mean(mse_scores)
                
            temporal_score = 1 - min(weighted_mse, 1.0)
            
            return {'temporal_score': temporal_score}
            
        except Exception as e:
            print(f"Error in temporal validation: {e}")
            return {'temporal_score': 0}
    
    def _complexity_penalty(self, features):
        """Calculate complexity penalty to discourage overfitting"""
        # Count derived features (non-base)
        base_features = [f for f in features if f in self.X_train.columns]
        derived_features = [f for f in features if f not in self.X_train.columns]
        
        # Calculate feature complexity
        n_derived = len(derived_features)
        n_total = len(features)
        
        # Optimal range is around 5-15 features
        size_penalty = 0
        if n_total < 5:
            size_penalty = 0.3  # Too few features
        elif n_total > 20:
            size_penalty = 0.2 * (n_total - 20) / 10  # Too many features
            
        # Penalize too many derived features
        derived_ratio = n_derived / max(n_total, 1)
        derived_penalty = 0 if derived_ratio <= 0.7 else (derived_ratio - 0.7) * 0.5
        
        # Calculate overall complexity penalty (lower is better)
        complexity_penalty = min(size_penalty + derived_penalty, 0.5)
        
        # Return inverted score (higher is better)
        return 1 - complexity_penalty
    
    def _feature_diversity_score(self, features):
        """Calculate feature diversity score to reward novel combinations"""
        # Compare current features with history
        if not self.feature_history:
            return 1.0  # First set is maximally diverse
            
        # Get all previously used features
        all_prev_features = set()
        for gen, feat_list in self.feature_history.items():
            all_prev_features.update(feat_list)
            
        # Calculate novelty - how many new features
        novel_features = [f for f in features if f not in all_prev_features]
        novelty_ratio = len(novel_features) / max(len(features), 1)
        
        # Calculate how different this is from the most recent generation
        recent_gen = max(self.feature_history.keys())
        recent_features = self.feature_history.get(recent_gen, [])
        
        overlap = set(features).intersection(set(recent_features))
        difference_ratio = 1 - len(overlap) / max(len(features) + len(recent_features) - len(overlap), 1)
        
        # Final diversity score 
        diversity_score = 0.7 * difference_ratio + 0.3 * novelty_ratio
        
        return diversity_score
    
    def _physical_validation(self, X_features, features):
        """Validate that feature relationships follow physical principles for rainfall"""
        try:
            # Train a simple model
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_features, self.y_train)
            
            # Get feature importances
            importances = model.feature_importances_
            
            # Verify key meteorological relationships
            weather_features = ['humidity', 'cloud', 'pressure', 'temparature', 'windspeed']
            present_weather = [f for f in weather_features if f in X_features.columns]
            
            if not present_weather:
                return 0.5  # Neutral if no weather features
            
            # Check expected correlations for available weather features
            score = 0.5  # Start neutral
            
            if 'humidity' in X_features.columns:
                humid_corr = np.corrcoef(X_features['humidity'], self.y_train)[0, 1]
                if humid_corr > 0:  # Humidity should positively correlate with rainfall
                    score += 0.1
                    
            if 'pressure' in X_features.columns:
                pressure_corr = np.corrcoef(X_features['pressure'], self.y_train)[0, 1]
                if pressure_corr < 0:  # Pressure often negatively correlates with rainfall
                    score += 0.1
            
            if 'cloud' in X_features.columns:
                cloud_corr = np.corrcoef(X_features['cloud'], self.y_train)[0, 1]
                if cloud_corr > 0:  # More clouds usually mean more rain
                    score += 0.1
                    
            # Cap the score at 1.0
            return min(score, 1.0)
            
        except Exception as e:
            print(f"Error in physical validation: {e}")
            return 0.5  # Neutral score on error
    
    def _calculate_fitness(self, scores):
        """Calculate overall fitness with component weights"""
        # Component weights - adjust these based on what's most important
        weights = {
            'cv_score': 0.35,          # Basic accuracy
            'temporal_score': 0.25,     # Time generalization
            'complexity_penalty': 0.15, # Avoid overfitting
            'diversity_score': 0.15,    # Novelty
            'physical_score': 0.10      # Domain knowledge
        }
        
        # Calculate weighted sum
        fitness = 0
        for component, weight in weights.items():
            if component in scores:
                fitness += scores[component] * weight
                
        # Cap fitness at 0.95 to avoid premature convergence
        return min(fitness, 0.95)
    
    def _update_feature_history(self, features, generation):
        """Track feature usage over generations"""
        if generation in self.feature_history:
            # Update existing generation
            self.feature_history[generation].extend(features)
            # Remove duplicates
            self.feature_history[generation] = list(set(self.feature_history[generation]))
        else:
            # New generation
            self.feature_history[generation] = list(set(features))
    
    def _print_validation_report(self, scores, features):
        """Print detailed validation report"""
        print("\n=== CHROMOSOME VALIDATION REPORT ===")
        print(f"Features: {len(features)}")
        print(f"MSE: {scores['mse_mean']:.4f} (±{scores['mse_std']:.4f})")
        print(f"CV Score: {scores['cv_score']:.4f}")
        print(f"Temporal Score: {scores['temporal_score']:.4f}")
        print(f"Complexity Score: {scores['complexity_penalty']:.4f}")
        print(f"Diversity Score: {scores['diversity_score']:.4f}")
        print(f"Physical Score: {scores['physical_score']:.4f}")
        print(f"Overall Fitness: {scores['fitness']:.4f}")
        print("===================================")
    
    def track_generation(self, generation, population, fitness_scores, best_chromosome):
        """Track statistics for a complete generation"""
        # Calculate average fitness
        avg_fitness = np.mean(fitness_scores)
        max_fitness = np.max(fitness_scores)
        
        # Store generation stats
        self.generation_scores.append({
            'generation': generation,
            'avg_fitness': avg_fitness,
            'max_fitness': max_fitness,
            'best_chromosome': best_chromosome.copy() if best_chromosome else None
        })
        
        # Plot progress every 5 generations
        if generation % 5 == 0 or generation == 1:
            self.plot_progress()
    
    def plot_progress(self):
        """Plot GA progress over generations"""
        if not self.generation_scores:
            return
            
        # Extract data
        generations = [g['generation'] for g in self.generation_scores]
        avg_fitness = [g['avg_fitness'] for g in self.generation_scores]
        max_fitness = [g['max_fitness'] for g in self.generation_scores]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Fitness over generations
        plt.subplot(2, 1, 1)
        plt.plot(generations, avg_fitness, 'b-', label='Average Fitness')
        plt.plot(generations, max_fitness, 'r-', label='Max Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('GA Fitness Progression')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Feature usage heatmap if we have enough generations
        if len(generations) >= 3:
            plt.subplot(2, 1, 2)
            
            # Get top features across generations
            all_features = set()
            for gen, features in self.feature_history.items():
                all_features.update(features)
            
            # Create feature usage matrix
            top_features = list(all_features)[:20]  # Limit to top 20
            feature_usage = np.zeros((len(generations), len(top_features)))
            
            for i, gen in enumerate(generations):
                if gen in self.feature_history:
                    gen_features = self.feature_history[gen]
                    for j, feature in enumerate(top_features):
                        feature_usage[i, j] = 1 if feature in gen_features else 0
            
            # Plot heatmap
            sns.heatmap(feature_usage, cmap='viridis', 
                       xticklabels=[f[:15] for f in top_features],
                       yticklabels=generations)
            plt.xlabel('Features')
            plt.ylabel('Generation')
            plt.title('Feature Usage Over Generations')
            
        plt.tight_layout()
        plt.savefig('analysis/ga_validation/ga_progress.png')
        plt.close()
        
        print(f"Saved progress plot to analysis/ga_validation/ga_progress.png")
    
    def final_validation(self, best_chromosome):
        """Comprehensive final validation of the best chromosome"""
        print("\n=== FINAL VALIDATION OF BEST CHROMOSOME ===")
        
        # Extract features
        features = self._extract_features(best_chromosome)
        print(f"Best chromosome has {len(features)} features")
        
        if not features:
            print("No valid features in best chromosome")
            return None
        
        # Apply features to full dataset
        X_features = self._apply_features(features)
        
        if X_features is None:
            print("Failed to create feature matrix for best chromosome")
            return None
            
        # Multi-fold CV with detailed reports
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []
        
        print("\nDetailed cross-validation:")
        for i, (train_idx, val_idx) in enumerate(kf.split(X_features)):
            # Split data
            X_train_fold = X_features.iloc[train_idx]
            y_train_fold = self.y_train.iloc[train_idx]
            X_val_fold = X_features.iloc[val_idx]
            y_val_fold = self.y_train.iloc[val_idx]
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42
            )
            
            model.fit(X_train_fold, y_train_fold)
            
            # Predict
            train_pred = model.predict(X_train_fold)
            val_pred = model.predict(X_val_fold)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train_fold, train_pred)
            val_mse = mean_squared_error(y_val_fold, val_pred)
            
            print(f"Fold {i+1}: Train MSE: {train_mse:.6f}, Val MSE: {val_mse:.6f}")
            fold_scores.append(val_mse)
            
            # Feature importance for this fold
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            print("  Top features:")
            for j in range(min(5, len(sorted_idx))):
                idx = sorted_idx[j]
                print(f"    {X_features.columns[idx]}: {importances[idx]:.4f}")
                
        # Overall score
        avg_mse = np.mean(fold_scores)
        std_mse = np.std(fold_scores)
        print(f"\nFinal validation MSE: {avg_mse:.6f} ±{std_mse:.6f}")
        print(f"Final score: {1-avg_mse:.6f}")
        
        # Save feature importance plot
        self._plot_feature_importance(X_features, features)
        
        return {
            'features': features,
            'mse': avg_mse,
            'score': 1 - avg_mse,
            'std': std_mse
        }
    
    def _plot_feature_importance(self, X_features, features):
        """Create feature importance visualization"""
        # Train on full dataset
        model = RandomForestRegressor(
            n_estimators=200, 
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        )
        
        model.fit(X_features, self.y_train)
        
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [X_features.columns[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('analysis/ga_validation/feature_importance.png')
        plt.close()
        
        print("Saved feature importance plot to analysis/ga_validation/feature_importance.png")