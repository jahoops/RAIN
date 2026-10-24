import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

def create_advanced_features(X):
    # 1. Polynomial interactions (degree 2)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    important_cols = X.columns[X.var() > X.var().quantile(0.75)]  # Focus on high-variance features
    poly_features = poly.fit_transform(X[important_cols])
    
    # 2. Statistical aggregations
    for col in important_cols:
        X[f'{col}_scaled'] = X[col] / X[col].mean()
        X[f'{col}_z'] = (X[col] - X[col].mean()) / X[col].std()
    
    # 3. Distance-based features
    X['distance_to_mean'] = np.sqrt(((X - X.mean())**2).sum(axis=1))
    
    # 4. Dimensionality reduction
    if X.shape[1] > 10:
        pca = PCA(n_components=5)
        pca_features = pca.fit_transform(X)
        for i in range(5):
            X[f'pca_{i}'] = pca_features[:, i]
    
    return X

