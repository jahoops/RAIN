import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import time

class LeaderboardAlignedValidator:
    """Validation class with a stricter evaluation scheme"""
    
    def __init__(self, X_train, y_train, temporal_column=None):
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.temporal_column = temporal_column
        self.feature_history = {}
        self.generation_scores = []
        
        # Create directory for plots
        os.makedirs("analysis/ga_validation", exist_ok=True)
        print(f"Initialized validator with {X_train.shape[1]} base features")
        
        # Calculate baseline MSE for scaling
        self._calculate_baseline()
    
    def _calculate_baseline(self):
        """Calculate baseline metrics to use for fitness scaling"""
        # Calculate baseline using simple models
        # 1. Mean baseline (always predict mean)
        mean_mse = mean_squared_error(self.y_train, 
                                      np.ones_like(self.y_train) * self.y_train.mean())
        
        # 2. Best single feature baseline
        best_single_mse = float('inf')
        for col in self.X_train.columns:
            if self.X_train[col].dtype.kind in 'fib' and self.X_train[col].isna().sum() == 0:
                try:
                    X = self.X_train[[col]]
                    
                    # Use 3-fold CV
                    cv = TimeSeriesSplit(n_splits=3)
                    mse_values = []
                    
                    for train_idx, val_idx in cv.split(X):
                        X_train_fold = X.iloc[train_idx]
                        y_train_fold = self.y_train.iloc[train_idx]
                        X_val_fold = X.iloc[val_idx]
                        y_val_fold = self.y_train.iloc[val_idx]
                        
                        model = RandomForestRegressor(n_estimators=50, random_state=42)
                        model.fit(X_train_fold, y_train_fold)
                        
                        val_pred = model.predict(X_val_fold)
                        mse = mean_squared_error(y_val_fold, val_pred)
                        mse_values.append(mse)
                    
                    # Use the average MSE
                    avg_mse = np.mean(mse_values)
                    if avg_mse < best_single_mse:
                        best_single_mse = avg_mse
                except:
                    continue
        
        # Store these for fitness scaling
        self.mean_baseline_mse = mean_mse
        self.best_single_mse = best_single_mse
        
        print(f"Baseline MSE (mean prediction): {mean_mse:.6f}")
        print(f"Baseline MSE (best single feature): {best_single_mse:.6f}")
        
        # Set target for a "good" model (50% better than best single feature)
        self.good_model_mse = best_single_mse * 0.5
        
        # Set target for "great" model (75% better than best single feature)
        self.great_model_mse = best_single_mse * 0.25
        
        print(f"Target good model MSE: {self.good_model_mse:.6f}")
        print(f"Target great model MSE: {self.great_model_mse:.6f}")
        
    def validate_chromosome(self, chromosome, generation=0, verbose=False):
        """Validate a chromosome using time-series validation with realistic scaling"""
        # Extract feature list
        features = self._extract_features(chromosome)
        
        if len(features) < 3:
            if verbose:
                print("Too few features to evaluate")
            return {'fitness': 0}
        
        try:
            # Apply features - use only existing columns
            X_features = self.X_train[features].copy()
            
            # Fill missing values
            for col in X_features.columns:
                missing_count = X_features[col].isna().sum()
                if missing_count > 0:
                    X_features[col] = X_features[col].fillna(X_features[col].median())
            
            # Time series validation with more folds
            tscv = TimeSeriesSplit(n_splits=5)
            errors = []
            
            # Apply k-fold time series validation
            for train_idx, val_idx in tscv.split(X_features):
                X_train_fold = X_features.iloc[train_idx]
                y_train_fold = self.y_train.iloc[train_idx]
                X_val_fold = X_features.iloc[val_idx]
                y_val_fold = self.y_train.iloc[val_idx]
                
                # Weight more recent folds more heavily
                model = RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=10,
                    min_samples_leaf=2,
                    random_state=42
                )
                
                model.fit(X_train_fold, y_train_fold)
                
                preds = model.predict(X_val_fold)
                mse = mean_squared_error(y_val_fold, preds)
                errors.append(mse)
            
            # Calculate weighted average of errors (most recent folds matter more)
            weights = [0.1, 0.15, 0.2, 0.25, 0.3][-len(errors):] 
            final_error = sum(e * w for e, w in zip(errors, weights[-len(errors):]))
            
            # Add penalty based on feature count - too few or too many features are penalized
            optimal_feature_count = 10
            feature_count_penalty = 0.1 * (abs(len(features) - optimal_feature_count)) / optimal_feature_count
            
            # Calculate scaled fitness - transformed to be more discriminating
            # Scale between baseline and great model performance
            normalized_error = (final_error - self.great_model_mse) / (self.mean_baseline_mse - self.great_model_mse)
            normalized_error = max(0, min(1, normalized_error))  # Clip to [0, 1]
            
            # Invert so higher is better and apply scaling to make differences more apparent
            raw_fitness = 1 - normalized_error
            
            # Apply a sigmoid-like scaling to spread out the middle values
            scaled_fitness = 1 / (1 + np.exp(-10 * (raw_fitness - 0.5)))
            
            # Apply feature penalty
            scaled_fitness = scaled_fitness * (1 - feature_count_penalty)
            
            generation_cap = min(0.6 + (generation * 0.02), 0.95)  # Gradually increase cap
            final_fitness = min(scaled_fitness, generation_cap)
            
            if verbose:
                print(f"\n===== Chromosome Evaluation =====")
                print(f"Features: {len(features)}")
                print(f"Fold MSEs: {', '.join([f'{e:.6f}' for e in errors])}")
                print(f"Weighted MSE: {final_error:.6f}")
                print(f"Feature count penalty: {feature_count_penalty:.4f}")
                print(f"Raw fitness: {raw_fitness:.4f}")
                print(f"Scaled fitness: {scaled_fitness:.4f}")
                print(f"Final fitness: {final_fitness:.4f}")
            
            return {'fitness': final_fitness, 'mse': final_error, 'features': features}
            
        except Exception as e:
            if verbose:
                print(f"Error in validation: {e}")
            return {'fitness': 0}
    
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
        
        # Store the best score for this generation
        self.generation_scores.append({
            'generation': generation,
            'avg_fitness': np.mean(fitness_scores) if fitness_scores else 0,
            'max_fitness': np.max(fitness_scores) if fitness_scores else 0
        })
        
        # Create visualization every 5 generations
        if generation % 5 == 0 or generation == 1:
            self._plot_progress()
            
        # Print detailed stats every generation
        top_fitness = sorted(fitness_scores, reverse=True)[:5]
        print(f"Generation {generation} stats:")
        print(f"  Top 5 fitness: {', '.join([f'{f:.4f}' for f in top_fitness])}")
        print(f"  Average fitness: {np.mean(fitness_scores):.4f}")
        print(f"  Fitness std dev: {np.std(fitness_scores):.4f}")
    
    def _plot_progress(self):
        """Plot GA progress"""
        if not self.generation_scores:
            return
        
        # Extract data
        generations = [g['generation'] for g in self.generation_scores]
        avg_fitness = [g['avg_fitness'] for g in self.generation_scores]
        max_fitness = [g['max_fitness'] for g in self.generation_scores]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(generations, avg_fitness, 'b-', label='Average Fitness')
        plt.plot(generations, max_fitness, 'r-', label='Max Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('GA Fitness Progress')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('analysis/ga_validation/ga_progress.png')
        plt.close()
    
    def final_validation(self, best_chromosome):
        """Perform final validation of best chromosome"""
        print("\n=== FINAL VALIDATION ===")
        
        # Extract features
        features = self._extract_features(best_chromosome)
        if not features:
            print("No valid features found in best chromosome")
            return {'fitness': 0}
            
        print(f"Best chromosome has {len(features)} features")
        
        # Get final evaluation with verbose output
        results = self.validate_chromosome(best_chromosome, verbose=True)
        
        # Create feature importance plot if possible
        try:
            self._plot_feature_importance(features)
        except Exception as e:
            print(f"Could not create feature importance plot: {e}")
            
        # Do a comprehensive K-fold validation as final check
        try:
            X_features = self.X_train[features].copy()
            for col in X_features.columns:
                if X_features[col].isna().sum() > 0:
                    X_features[col] = X_features[col].fillna(X_features[col].median())
                    
            # Use 10-fold CV for final evaluation
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            cv_scores = []
            
            for train_idx, val_idx in kf.split(X_features):
                X_train_fold = X_features.iloc[train_idx]
                y_train_fold = self.y_train.iloc[train_idx]
                X_val_fold = X_features.iloc[val_idx]
                y_val_fold = self.y_train.iloc[val_idx]
                
                model = RandomForestRegressor(n_estimators=200, random_state=42)
                model.fit(X_train_fold, y_train_fold)
                
                preds = model.predict(X_val_fold)
                mse = mean_squared_error(y_val_fold, preds)
                cv_scores.append(mse)
                
            print(f"\nFinal 10-fold CV MSE: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
            print(f"Estimated score: {1 - np.mean(cv_scores):.6f}")
                
        except Exception as e:
            print(f"Error in final validation: {e}")
        
        return results
    
    def _plot_feature_importance(self, features):
        """Plot feature importance"""
        X_features = self.X_train[features]
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_features, self.y_train)
        
        # Get importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [features[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('analysis/ga_validation/feature_importance.png')
        plt.close()
        
        print("Saved feature importance plot")
        
        # Print top 10 features
        print("\nTop 10 features by importance:")
        for i in range(min(10, len(indices))):
            idx = indices[i]
            print(f"{i+1}. {features[idx]}: {importances[idx]:.6f}")