import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from itertools import combinations
from tqdm import tqdm
import time
import pickle
import os

class MonteCarloFeatureDiscovery:
    def __init__(self, X_train, y_train, base_features, n_iterations=1000, test_size=0.2):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train, y_train, test_size=test_size, random_state=42)
        self.base_features = list(base_features)
        self.n_iterations = n_iterations
        self.results = []
        self.best_score = 0
        self.best_features = []
        
        # Create a directory to store results
        os.makedirs('feature_engineering', exist_ok=True)
        
    def _evaluate_feature_set(self, features):
        """Train a model and evaluate a set of features"""
        # Use RandomForest with limited trees for fast evaluation
        model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        model.fit(self.X_train[features], self.y_train)
        y_pred = model.predict(self.X_val[features])
        score = 1 - mean_squared_error(self.y_val, y_pred)
        return score
    
    def _generate_transformations(self, feature):
        """Generate mathematical transformations of a feature"""
        transformations = {}
        
        # Skip transformations for categorical or binary features
        if feature in ['day']:
            return transformations
        
        # Create transformations
        x = self.X_train[feature]
        
        # Skip if feature has zeros or negative values
        if (x <= 0).any():
            return transformations
        
        # Logarithmic
        transformations[f'log_{feature}'] = np.log1p(x)
        
        # Square root
        transformations[f'sqrt_{feature}'] = np.sqrt(x)
        
        # Squared
        transformations[f'square_{feature}'] = x ** 2
        
        # Cube
        transformations[f'cube_{feature}'] = x ** 3
        
        # Inverse
        if (x.abs() > 0.001).all():
            transformations[f'inverse_{feature}'] = 1 / x
        
        return transformations
    
    def _generate_interactions(self, features, max_interactions=2):
        """Generate interaction features between selected features"""
        interactions = {}
        
        if len(features) < 2:
            return interactions
        
        # Create pairwise interactions
        for feat1, feat2 in list(combinations(features, 2))[:max_interactions]:
            # Skip interactions with categorical features
            if feat1 in ['day'] or feat2 in ['day']:
                continue
                
            # Multiplication
            interactions[f'{feat1}_x_{feat2}'] = self.X_train[feat1] * self.X_train[feat2]
            
            # Division (if no zeros)
            if (self.X_train[feat2].abs() > 0.001).all():
                interactions[f'{feat1}_div_{feat2}'] = self.X_train[feat1] / self.X_train[feat2]
                
            # Sum
            interactions[f'{feat1}_plus_{feat2}'] = self.X_train[feat1] + self.X_train[feat2]
            
            # Difference
            interactions[f'{feat1}_minus_{feat2}'] = self.X_train[feat1] - self.X_train[feat2]
        
        return interactions
    
    def _add_temporal_features(self, features):
        """Add specialized temporal features"""
        temporal = {}
        
        # Identify rainfall related features
        rain_features = [f for f in features if 'rain' in f]
        
        if not rain_features:
            return temporal
        
        # Extract a rainfall feature as base
        rain_feat = rain_features[0]
        
        # Create rolling difference from the base feature
        if 'rainfall_lag1' in features and rain_feat != 'rainfall_lag1':
            temporal[f'{rain_feat}_change'] = self.X_train[rain_feat] - self.X_train['rainfall_lag1']
        
        return temporal
    
    def run_discovery(self):
        """Run the Monte Carlo feature discovery process"""
        print("Starting Monte Carlo feature discovery...")
        
        # Track progress
        start_time = time.time()
        
        for iteration in tqdm(range(self.n_iterations)):
            # 1. Randomly select a subset of base features
            n_features = random.randint(3, len(self.base_features))
            selected_features = random.sample(self.base_features, n_features)
            
            # 2. Add transformations for some features
            transformations = {}
            for feat in selected_features[:3]:  # Limit to 3 transformations for efficiency
                transformations.update(self._generate_transformations(feat))
                
            # 3. Add interactions between features
            interactions = self._generate_interactions(selected_features)
                
            # 4. Add temporal features
            temporal_features = self._add_temporal_features(selected_features)
            
            # 5. Create feature matrix for this iteration
            iteration_features = list(selected_features)
            
            # Add the new features to the dataframe temporarily
            temp_df = self.X_train.copy()
            
            for name, values in transformations.items():
                temp_df[name] = values
                iteration_features.append(name)
                
            for name, values in interactions.items():
                temp_df[name] = values
                iteration_features.append(name)
                
            for name, values in temporal_features.items():
                temp_df[name] = values
                iteration_features.append(name)
            
            # 6. Evaluate the feature set
            try:
                score = self._evaluate_feature_set(iteration_features)
                
                # Store the result
                self.results.append({
                    'iteration': iteration,
                    'score': score,
                    'n_features': len(iteration_features),
                    'features': iteration_features
                })
                
                # Update best score
                if score > self.best_score:
                    self.best_score = score
                    self.best_features = iteration_features
                    print(f"\nNew best score: {score:.6f} with {len(iteration_features)} features")
                    print(f"Top features: {iteration_features[:5]}...")
                    
                    # Save the best result so far
                    with open('feature_engineering/best_feature_set.pkl', 'wb') as f:
                        pickle.dump({
                            'score': score,
                            'features': iteration_features
                        }, f)
            except Exception as e:
                # Skip errors (e.g., from invalid transformations)
                pass
                
            # 7. Periodically save results
            if iteration % 100 == 0:
                results_df = pd.DataFrame(self.results)
                results_df.to_csv('feature_engineering/monte_carlo_results.csv', index=False)
                
                # Also save the transformations and interactions for the best feature set
                if self.best_features:
                    best_transformations = [f for f in self.best_features if f not in self.base_features]
                    with open('feature_engineering/best_transformations.txt', 'w') as f:
                        f.write('\n'.join(best_transformations))
        
        # Save final results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv('feature_engineering/monte_carlo_final_results.csv', index=False)
        
        # Print summary
        elapsed_time = time.time() - start_time
        print(f"\nDiscovery completed in {elapsed_time:.2f} seconds")
        print(f"Best score: {self.best_score:.6f}")
        print(f"Best feature count: {len(self.best_features)}")
        print(f"Top 10 best features: {self.best_features[:10]}")
        
        return self.best_features, self.best_score
    
    def apply_best_features(self, X_test):
        """Apply the best feature set to test data"""
        # Create the same transformations for test data
        test_df = X_test.copy()
        
        # Only include base features that are in our best set
        new_features = [f for f in self.best_features if f in self.base_features]
        
        # Apply transformations
        for feature in self.best_features:
            if '_x_' in feature:
                feat1, feat2 = feature.split('_x_')
                test_df[feature] = test_df[feat1] * test_df[feat2]
            elif '_div_' in feature:
                feat1, feat2 = feature.split('_div_')
                test_df[feature] = test_df[feat1] / test_df[feat2]
            elif '_plus_' in feature:
                feat1, feat2 = feature.split('_plus_')
                test_df[feature] = test_df[feat1] + test_df[feat2]
            elif '_minus_' in feature:
                feat1, feat2 = feature.split('_minus_')
                test_df[feature] = test_df[feat1] - test_df[feat2]
            elif 'log_' in feature:
                base_feat = feature.replace('log_', '')
                test_df[feature] = np.log1p(test_df[base_feat])
            elif 'sqrt_' in feature:
                base_feat = feature.replace('sqrt_', '')
                test_df[feature] = np.sqrt(test_df[base_feat])
            elif 'square_' in feature:
                base_feat = feature.replace('square_', '')
                test_df[feature] = test_df[base_feat] ** 2
            elif 'cube_' in feature:
                base_feat = feature.replace('cube_', '')
                test_df[feature] = test_df[base_feat] ** 3
            elif 'inverse_' in feature:
                base_feat = feature.replace('inverse_', '')
                test_df[feature] = 1 / test_df[base_feat]
            elif '_change' in feature:
                base_feat = feature.replace('_change', '')
                if 'rainfall_lag1' in test_df.columns:
                    test_df[feature] = test_df[base_feat] - test_df['rainfall_lag1']
        
        return test_df

# Usage example
if __name__ == "__main__":
    from data_pipeline import create_complete_data_pipeline
    
    # Load data
    X_train_temporal, X_test_temporal = create_complete_data_pipeline()
    y_train = X_train_temporal['rainfall']
    X_train = X_train_temporal.drop(['id', 'rainfall'], axis=1)
    X_test = X_test_temporal.drop(['id'], axis=1)
    
    # Run Monte Carlo discovery
    mc_discovery = MonteCarloFeatureDiscovery(
        X_train, y_train, X_train.columns, n_iterations=1000
    )
    best_features, best_score = mc_discovery.run_discovery()
    
    # Apply to test data and train final model
    X_test_transformed = mc_discovery.apply_best_features(X_test)
    
    # Train final model on best features
    final_model = RandomForestRegressor(
        n_estimators=1000, max_depth=None, random_state=87, n_jobs=-1
    )
    final_model.fit(X_train[best_features], y_train)
    
    # Generate predictions
    predictions = final_model.predict(X_test_transformed[best_features])
    
    # Create submission
    submission = pd.DataFrame({
        'id': X_test_temporal['id'],
        'rainfall': predictions
    })
    submission.to_csv('/home/USER/Projects/RAIN/data/candidates/monte_carlo_features.csv', index=False)
    print("Created submission with Monte Carlo discovered features")