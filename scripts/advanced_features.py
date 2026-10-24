import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler  # Add this import
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_regression

def create_advanced_features(train_df, test_df):
    """Create sophisticated feature transformations"""
    print("Creating advanced features...")
    
    # Store IDs
    train_ids = train_df['id']
    test_ids = test_df['id']
    target = None
    if 'rainfall' in train_df.columns:
        target = train_df['rainfall']
    
    # Remove ID and target columns
    X_train = train_df.drop(['id'], axis=1)
    if 'rainfall' in X_train.columns:
        X_train = X_train.drop(['rainfall'], axis=1)
        
    X_test = test_df.drop(['id'], axis=1)
    
    # Make sure we're only using common columns
    common_cols = list(set(X_train.columns).intersection(set(X_test.columns)))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    # Create a combined dataset for transformations
    combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    n_train = len(X_train)
    
    # Save original feature count for comparison
    original_features = combined.shape[1]
    print(f"Original feature count: {original_features}")
    
    # 1. RBF features for numeric columns
    numeric_cols = combined.select_dtypes(include=['float64', 'int64']).columns
    centers = np.array([0.1, 0.5, 0.9])
    
    for col in numeric_cols:
        col_min = combined[col].min()
        col_max = combined[col].max()
        if col_max > col_min:  # Avoid division by zero
            normalized = (combined[col] - col_min) / (col_max - col_min)
            for center in centers:
                gamma = 10  # RBF width parameter
                combined[f'{col}_rbf_{center:.1f}'] = np.exp(-gamma * (normalized - center)**2)
    
    # 2. PCA features
    try:
        scaler = StandardScaler()
        scaled_numeric = scaler.fit_transform(combined[numeric_cols])
        
        pca = PCA(n_components=min(10, len(numeric_cols)))
        pca_features = pca.fit_transform(scaled_numeric)
        
        for i in range(pca_features.shape[1]):
            combined[f'pca_{i+1}'] = pca_features[:, i]
            
        print(f"Added {pca_features.shape[1]} PCA components")
        
        # Add explained variance as a feature
        explained_variance = pca.explained_variance_ratio_
        for i in range(min(5, len(explained_variance))):
            combined[f'pca_var_{i+1}'] = explained_variance[i]
    except:
        print("Could not create PCA features")
    
    # 3. Feature crosses (polynomial features)
    try:
        for i, col1 in enumerate(numeric_cols[:10]):  # Limit to avoid explosion
            combined[f'{col1}_squared'] = combined[col1] ** 2
            combined[f'{col1}_cubed'] = combined[col1] ** 3
            
            for j, col2 in enumerate(numeric_cols[:10]):
                if i < j:  # Avoid duplicates
                    combined[f'{col1}_times_{col2}'] = combined[col1] * combined[col2]
                    combined[f'{col1}_plus_{col2}'] = combined[col1] + combined[col2]
                    combined[f'{col1}_minus_{col2}'] = combined[col1] - combined[col2]
                    
                    # Ratios with protection against division by zero
                    denominator = combined[col2].copy()
                    denominator[denominator == 0] = 1e-5
                    combined[f'{col1}_div_{col2}'] = combined[col1] / denominator
    except:
        print("Could not create all feature crosses")
    
    # 4. Clustering-based features
    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(scaled_numeric)
        combined['kmeans_cluster'] = clusters
        
        # Distance to cluster centers
        for i, center in enumerate(kmeans.cluster_centers_):
            distances = np.linalg.norm(scaled_numeric - center, axis=1)
            combined[f'dist_cluster_{i}'] = distances
    except:
        print("Could not create clustering features")
    
    # 5. Feature aggregations - group features by their correlation patterns
    if target is not None:
        corr_groups = {}
        X_with_target = X_train.copy()
        X_with_target['target'] = target
        
        for col in numeric_cols:
            corr = np.corrcoef(X_with_target[col], X_with_target['target'])[0, 1]
            group = int(np.sign(corr) * np.floor(abs(corr) * 5))  # Group by correlation strength and sign
            if group not in corr_groups:
                corr_groups[group] = []
            corr_groups[group].append(col)
        
        # Create aggregation features for each group
        for group, cols in corr_groups.items():
            if len(cols) > 1:
                combined[f'group_{group}_mean'] = combined[cols].mean(axis=1)
                combined[f'group_{group}_std'] = combined[cols].std(axis=1)
                combined[f'group_{group}_max'] = combined[cols].max(axis=1)
                combined[f'group_{group}_min'] = combined[cols].min(axis=1)
    
    # Handle potential infinities and NaNs
    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill NaNs with column medians
    for col in combined.columns:
        if combined[col].isna().any():
            median = combined[col].median()
            combined[col].fillna(median if not pd.isna(median) else 0, inplace=True)
    
    # Split back into train and test
    X_train_enhanced = combined.iloc[:n_train].copy()
    X_test_enhanced = combined.iloc[n_train:].copy()
    
    # Feature selection if we have the target
    if target is not None:
        try:
            # Select top K features based on mutual information
            selector = SelectKBest(mutual_info_regression, k=min(500, X_train_enhanced.shape[1]))
            selector.fit(X_train_enhanced, target)
            
            # Get selected feature mask and names
            selected_features = X_train_enhanced.columns[selector.get_support()]
            
            print(f"Selected {len(selected_features)} best features out of {X_train_enhanced.shape[1]}")
            
            # Keep only selected features
            X_train_enhanced = X_train_enhanced[selected_features]
            X_test_enhanced = X_test_enhanced[selected_features]
        except:
            print("Feature selection failed, using all features")
    
    # Add back IDs
    X_train_enhanced['id'] = train_ids
    X_test_enhanced['id'] = test_ids
    
    # Add back target if it exists
    if target is not None:
        X_train_enhanced['rainfall'] = target
    
    print(f"Final feature count: {X_train_enhanced.shape[1] - 1}")  # -1 for ID column
    print(f"Added {X_train_enhanced.shape[1] - original_features - 1} new features")
    
    return X_train_enhanced, X_test_enhanced

if __name__ == "__main__":
    from data_loader import load_data, preprocess_data
    
    # Load data
    train_df, test_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # Create advanced features
    train_enhanced, test_enhanced = create_advanced_features(train_df, test_df)
    
    # Save enhanced datasets
    train_enhanced.to_csv('data/train_advanced_features.csv', index=False)
    test_enhanced.to_csv('data/test_advanced_features.csv', index=False)
    
    print("Enhanced datasets saved to data/train_advanced_features.csv and data/test_advanced_features.csv")
