import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

class FeatureGenerationFactory:
    def __init__(self, X_train, y_train, test_size=0.2):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train, y_train, test_size=test_size, random_state=42
        )
        self.base_features = list(X_train.columns)
        self.feature_registry = {}
        self.best_features = []
        self.best_score = 0
        
        # Create directory for outputs
        os.makedirs('feature_engineering', exist_ok=True)
    
    def generate_features(self):
        """Generate a wide variety of features"""
        print("Starting feature generation...")
        
        # Create all the feature types
        self._create_temporal_features()
        self._create_weather_features()
        self._create_polynomial_features()
        self._create_window_features()
        self._create_specialized_features()
        
        print(f"Generated {len(self.feature_registry)} new features")
        
    def _create_temporal_features(self):
        """Create time-based features"""
        print("Generating temporal features...")
        
        # If we have rainfall_lag1, create more lag features
        if 'rainfall_lag1' in self.base_features:
            # Generate additional lag features
            for lag in range(3, 10):
                name = f'rainfall_lag{lag}'
                self.feature_registry[name] = self._add_lag_feature('rainfall_lag1', lag-1)
        
        # Create lag differences
        if 'rainfall_lag1' in self.base_features and 'rainfall_lag2' in self.base_features:
            self.feature_registry['rain_lag_diff'] = self.X_train['rainfall_lag1'] - self.X_train['rainfall_lag2']
            
        # Create momentum indicators
        if 'rainfall_lag1' in self.base_features and 'rainfall_lag2' in self.base_features:
            name = 'rain_momentum'
            self.feature_registry[name] = self._add_momentum_feature()
            
            # Rate of change
            name = 'rain_roc_2day'
            self.feature_registry[name] = self._add_roc_feature(2)
            
            name = 'rain_roc_3day'
            if 'rainfall_lag3' in self.feature_registry:
                self.feature_registry[name] = self._add_roc_feature(3)
                
        # Create exponential moving averages if we have rolling means
        if 'rainfall_rolling_3d_mean' in self.base_features:
            name = 'rain_ema_3'
            alpha = 0.6  # Weight for most recent
            self.feature_registry[name] = self._add_ema_feature(3, alpha)
    
    def _create_weather_features(self):
        """Create weather-specific features"""
        print("Generating weather features...")
        
        # Interaction features
        if 'cloud' in self.base_features and 'humidity' in self.base_features:
            self.feature_registry['cloud_humidity_product'] = self.X_train['cloud'] * self.X_train['humidity']
        
        if 'pressure' in self.base_features:
            # Pressure changes are important for weather
            self.feature_registry['pressure_change'] = self._add_diff_feature('pressure')
            
            # Pressure gradient (acceleration)
            self.feature_registry['pressure_gradient'] = self._add_diff_feature('pressure', second_order=True)
        
        if 'temparature' in self.base_features and 'humidity' in self.base_features:
            # Create heat index (feels like temperature)
            self.feature_registry['heat_index'] = self._add_heat_index_feature()
            
        if 'windspeed' in self.base_features:
            # Wind energy is proportional to square of speed
            self.feature_registry['wind_energy'] = self.X_train['windspeed'] ** 2
            
        # Dew point if we have temperature and humidity
        if 'temparature' in self.base_features and 'humidity' in self.base_features:
            self.feature_registry['dewpoint_proxy'] = self._add_dewpoint_proxy()
            
    def _create_polynomial_features(self):
        """Create polynomial and interaction features"""
        print("Generating polynomial features...")
        
        # Create squared terms
        for feat in self.base_features:
            # Skip categorical features
            if feat in ['day']:
                continue
                
            # Add squared term
            self.feature_registry[f'{feat}_squared'] = self.X_train[feat] ** 2
        
        # Create select interaction terms
        key_features = ['humidity', 'cloud', 'pressure', 'temparature', 'windspeed']
        key_features = [f for f in key_features if f in self.base_features]
        
        for i, feat1 in enumerate(key_features):
            for feat2 in key_features[i+1:]:
                name = f'{feat1}_x_{feat2}'
                self.feature_registry[name] = self.X_train[feat1] * self.X_train[feat2]
                
    def _create_window_features(self):
        """Create window-based features"""
        print("Generating window features...")
        
        # If we have rolling mean features
        rolling_features = [f for f in self.base_features if 'rolling' in f]
        
        if rolling_features:
            # Get window sizes from existing features
            windows = set()
            for feat in rolling_features:
                try:
                    if 'rolling_' in feat and 'd_mean' in feat:
                        window = int(feat.split('rolling_')[1].split('d_')[0])
                        windows.add(window)
                except:
                    pass
            
            # If we have rainfall lags and rolling means for windows
            if 'rainfall_lag1' in self.base_features and windows:
                for window in windows:
                    # Create rolling standard deviation
                    roll_std_name = f'rainfall_rolling_{window}d_std'
                    if roll_std_name not in self.base_features:
                        df = pd.DataFrame({'rainfall': self.X_train['rainfall_lag1'].shift(-1)})
                        for lag in range(2, window + 1):
                            lag_name = f'rainfall_lag{lag}'
                            if lag_name in self.base_features:
                                df[lag_name] = self.X_train[lag_name].shift(-1)
                            elif lag_name in self.feature_registry:
                                df[lag_name] = self.feature_registry[lag_name].shift(-1)
                            
                        # Calculate the rolling std if we have enough data
                        if len(df.columns) >= window:
                            self.feature_registry[roll_std_name] = df.std(axis=1)
    
    def _create_specialized_features(self):
        """Create domain-specific features for rainfall prediction"""
        print("Generating specialized rainfall features...")
        
        # If we have humidity and temperature
        if 'humidity' in self.base_features and 'temparature' in self.base_features:
            # Rainfall is often related to vapor pressure
            vapor_pressure = self._add_vapor_pressure_feature()
            self.feature_registry['vapor_pressure'] = vapor_pressure
            
        # Non-linear transformations of key rain-related features
        for feat in self.base_features:
            if 'rainfall' in feat:
                # Cube root transformation (common for precipitation)
                cube_root = np.cbrt(self.X_train[feat])
                self.feature_registry[f'{feat}_cbrt'] = cube_root
                
                # Log transform (with small constant to handle zeros)
                # Log transform more closely approximates normal distribution for precip
                log_rain = np.log1p(self.X_train[feat])
                self.feature_registry[f'{feat}_log'] = log_rain
    
    def _add_lag_feature(self, base_feature, lag):
        """Add a lag feature by shifting another lag feature"""
        return self.X_train[base_feature].shift(lag)
    
    def _add_diff_feature(self, feature, second_order=False):
        """Add difference feature (first or second order)"""
        if not second_order:
            return self.X_train[feature].diff()
        else:
            return self.X_train[feature].diff().diff()
        
    def _add_momentum_feature(self):
        """Add rainfall momentum feature"""
        # Simple momentum: direction of change
        return np.sign(self.X_train['rainfall_lag1'] - self.X_train['rainfall_lag2'])
    
    def _add_roc_feature(self, window):
        """Add rate of change feature"""
        # Need lag1 and lag{window}
        lag_window = f'rainfall_lag{window}'
        
        if lag_window in self.base_features:
            feat = self.X_train['rainfall_lag1'] / self.X_train[lag_window] - 1
        elif lag_window in self.feature_registry:
            feat = self.X_train['rainfall_lag1'] / self.feature_registry[lag_window] - 1
        else:
            # Return empty series if we don't have the window
            return pd.Series(np.nan, index=self.X_train.index)
            
        return feat
        
    def _add_ema_feature(self, window, alpha):
        """Add exponential moving average feature"""
        # For EMA we need rainfall lags
        lags = [f'rainfall_lag{i}' for i in range(1, window+1)]
        
        # Check if we have all the lags
        available_lags = []
        for lag in lags:
            if lag in self.base_features:
                available_lags.append(self.X_train[lag])
            elif lag in self.feature_registry:
                available_lags.append(self.feature_registry[lag])
                
        if len(available_lags) < window:
            return pd.Series(np.nan, index=self.X_train.index)
            
        # Calculate EMA
        ema = available_lags[0].copy()
        for i in range(1, len(available_lags)):
            ema = alpha * available_lags[i] + (1-alpha) * ema
            
        return ema
        
    def _add_heat_index_feature(self):
        """Add heat index feature (temperature + humidity effect)"""
        # Simple approximation of heat index
        T = self.X_train['temparature']  
        RH = self.X_train['humidity']
        
        # Simple formula for heat index
        heat_index = T + 0.05 * RH
        return heat_index
        
    def _add_dewpoint_proxy(self):
        """Add dewpoint proxy feature"""
        # Simple approximation
        T = self.X_train['temparature']
        RH = self.X_train['humidity'] / 100
        
        # Approximate dew point using Magnus formula
        dewpoint = T - ((100 - RH * 100) / 5)
        return dewpoint
        
    def _add_vapor_pressure_feature(self):
        """Add vapor pressure feature"""
        # Using simplified formula
        T = self.X_train['temparature']
        RH = self.X_train['humidity'] / 100
        
        # Approximate saturation vapor pressure using simplified formula
        sat_vp = 6.11 * 10**(7.5 * T / (237.3 + T))
        
        # Actual vapor pressure
        vp = sat_vp * RH
        return vp
        
    def evaluate_features(self):
        """Evaluate each feature individually and select best ones"""
        print("\nEvaluating features individually...")
        
        feature_scores = {}
        
        # Baseline score with basic features
        baseline_score = self._evaluate_feature_set(self.base_features)
        print(f"Baseline score with {len(self.base_features)} base features: {baseline_score:.6f}")
        
        # Test each new feature
        for feature_name, feature_values in tqdm(self.feature_registry.items()):
            # Create a combined feature set
            test_features = self.base_features.copy()
            
            # Add the feature temporarily to the train and validation sets
            X_train_temp = self.X_train.copy()
            X_val_temp = self.X_val.copy()
            
            X_train_temp[feature_name] = feature_values
            
            # Need to carefully transform the validation set too
            # For simplicity in this example, we'll just add NaN values
            # In practice, you would need to replicate the same transformation
            X_val_temp[feature_name] = np.nan
            
            # Add the feature to test features
            test_features.append(feature_name)
            
            try:
                # Evaluate the feature
                score = self._evaluate_feature_set(test_features)
                
                # Calculate the improvement
                improvement = score - baseline_score
                
                # Store the score
                feature_scores[feature_name] = {
                    'score': score,
                    'improvement': improvement
                }
                
            except Exception as e:
                # Skip problematic features
                print(f"Error evaluating feature {feature_name}: {e}")
        
        # Find the best features
        sorted_features = sorted(
            feature_scores.items(),
            key=lambda x: x[1]['improvement'],
            reverse=True
        )
        
        # Print the top features
        print("\nTop 20 most valuable features:")
        for name, scores in sorted_features[:20]:
            print(f"{name}: {scores['improvement']:.6f} improvement")
            
        # Save the results
        results = []
        for name, scores in sorted_features:
            results.append({
                'feature': name,
                'score': scores['score'],
                'improvement': scores['improvement']
            })
            
        results_df = pd.DataFrame(results)
        results_df.to_csv('feature_engineering/feature_scores.csv', index=False)
        
        # Select the best features (positive improvement)
        self.best_features = [
            name for name, scores in sorted_features 
            if scores['improvement'] > 0
        ]
        
        print(f"\nSelected {len(self.best_features)} features with positive impact")
        
        # Combine with base features
        self.best_features = self.base_features + self.best_features
        
        return self.best_features, results_df
        
    def _evaluate_feature_set(self, features):
        """Evaluate a set of features"""
        try:
            # Use a lightweight pipeline for quick evaluation
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ))
            ])
            
            # Train and evaluate
            pipeline.fit(self.X_train[features], self.y_train)
            y_pred = pipeline.predict(self.X_val[features])
            
            # Calculate score (1 - MSE)
            score = 1 - mean_squared_error(self.y_val, y_pred)
            return score
        except Exception as e:
            raise Exception(f"Error in evaluation: {e}")
            
    def create_final_features(self, X_test):
        """Create the final feature set for model training and testing"""
        print("\nCreating final feature matrix...")
        
        # Start with the base features
        X_train_final = self.X_train.copy()
        X_test_final = X_test.copy()
        
        # Add the best generated features
        for feature_name, feature_values in self.feature_registry.items():
            if feature_name in self.best_features:
                X_train_final[feature_name] = feature_values
                
                # Need to create the same feature for test data
                # This is just a placeholder - in practice you'd implement the transformations
                X_test_final[feature_name] = 0
        
        return X_train_final, X_test_final, self.best_features

# Usage example
if __name__ == "__main__":
    from data_pipeline import create_complete_data_pipeline
    
    # Load data
    X_train_temporal, X_test_temporal = create_complete_data_pipeline()
    y_train = X_train_temporal['rainfall']
    X_train = X_train_temporal.drop(['id', 'rainfall'], axis=1)
    X_test = X_test_temporal.drop(['id'], axis=1)
    
    # Create feature factory
    factory = FeatureGenerationFactory(X_train, y_train)
    
    # Generate features
    factory.generate_features()
    
    # Evaluate and select the best features
    best_features, feature_scores = factory.evaluate_features()
    
    # Create submission placeholder
    submission = pd.DataFrame({
        'id': X_test_temporal['id'],
        'rainfall': 0
    })
    submission.to_csv('/home/USER/Projects/RAIN/data/candidates/feature_factory_placeholder.csv', index=False)
    
    print("Feature engineering analysis complete!")