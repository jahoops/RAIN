import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFE
import os
from optimized_binary_model import create_binary_optimized_features
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV

def create_enhanced_logistic_submission():
    """Create a submission using enhanced logistic regression techniques"""
    # Load data
    train = pd.read_csv('/home/USER/Projects/RAIN/data/train.csv')
    test = pd.read_csv('/home/USER/Projects/RAIN/data/test.csv')

    # Check if date column exists or find appropriate time column
    date_column = None
    potential_date_columns = ['date', 'time', 'datetime', 'timestamp', 'day', 'id']
    
    for col in potential_date_columns:
        if col in train.columns:
            print(f"Found potential time column: {col}")
            date_column = col
            break
    
    # If we found a date/time column, try to use it
    if (date_column and date_column != 'id'):
        try:
            # Try to convert to datetime
            train[date_column] = pd.to_datetime(train[date_column])
            test[date_column] = pd.to_datetime(test[date_column])
            
            # Create temporal features
            train['month'] = train[date_column].dt.month
            train['day_of_year'] = train[date_column].dt.dayofyear
            test['month'] = test[date_column].dt.month
            test['day_of_year'] = test[date_column].dt.dayofyear
            
            print(f"Created temporal features from {date_column}")
        except:
            print(f"Could not convert {date_column} to datetime. Skipping temporal features.")
    elif date_column == 'id':
        # If only ID is available, it might be sequential and represent time
        print("Using ID as a proxy for time sequence")
        train['time_proxy'] = train[date_column] / max(train[date_column])
        test['time_proxy'] = test[date_column] / max(test[date_column])
    else:
        print("No datetime column found. Skipping temporal features.")

    # Store the datetime columns if created
    datetime_cols = []
    if date_column and date_column != 'id':
        datetime_cols = [date_column]
    
    # Remove datetime columns before feature creation
    if datetime_cols:
        print(f"Temporarily removing datetime columns: {datetime_cols}")
        train_feature = train.drop(columns=datetime_cols)
        test_feature = test.drop(columns=datetime_cols)
    else:
        train_feature = train
        test_feature = test
    
    # Generate features
    X, X_test = create_binary_optimized_features(train_feature, test_feature)
    y = train['rainfall']
    
    # Add back any numeric temporal features we created
    for col in ['month', 'day_of_year', 'time_proxy']:
        if col in train.columns:
            X[col] = train[col]
        if col in test.columns:
            X_test[col] = test[col]
    
    # Create the same features for both training and test sets
    print("Creating additional meteorological features...")
    
    # Humidity-temperature interaction (warm & humid vs. cool & humid)
    X['humid_warm'] = X['humidity'] * X['maxtemp'] / 100
    X_test['humid_warm'] = X_test['humidity'] * X_test['maxtemp'] / 100
    
    # Pressure gradient (with NA handling)
    X['pressure_change'] = X['pressure'] - X['pressure'].shift(1)
    X['pressure_change'] = X['pressure_change'].fillna(0)  # Fill NA for first row
    
    X_test['pressure_change'] = X_test['pressure'] - X_test['pressure'].shift(1)
    X_test['pressure_change'] = X_test['pressure_change'].fillna(0)  # Fill NA for first row
    
    # Dew point depression (temperature - dewpoint)
    if 'dewpoint' in X.columns and 'temparature' in X.columns:
        X['dewpoint_depression'] = X['temparature'] - X['dewpoint']
        X_test['dewpoint_depression'] = X_test['temparature'] - X_test['dewpoint']
    
    # Cloud-sunshine interaction
    if 'cloud' in X.columns and 'sunshine' in X.columns:
        X['cloud_sunshine'] = X['cloud'] * (1 - X['sunshine']/100)
        X_test['cloud_sunshine'] = X_test['cloud'] * (1 - X_test['sunshine']/100)
    
    # Note that we're NOT using SMOTE here - it can sometimes hurt performance
    # with logistic regression due to creating synthetic points
    
    preprocessing = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Scale features - helps with logistic regression convergence
    column_names = X.columns
    X = preprocessing.fit_transform(X)
    X_scaled = pd.DataFrame(X, columns=column_names)
    X_test_scaled = pd.DataFrame(preprocessing.transform(X_test), columns=column_names)
    
    print(f"Training enhanced logistic regression with {X_scaled.shape[1]} features")
    
    # Strategy 1: Use Recursive Feature Elimination instead of SelectFromModel
    print("Performing Recursive Feature Elimination...")
    base_lr = LogisticRegression(C=0.5, max_iter=10000, solver='liblinear', random_state=42)
    rfe = RFE(estimator=base_lr, n_features_to_select=20, step=1)
    rfe.fit(X_scaled, y)
    selected_features_rfe = X_scaled.columns[rfe.support_]
    
    print(f"RFE selected {len(selected_features_rfe)} features")
    X_rfe = X_scaled[selected_features_rfe]
    X_test_rfe = X_test_scaled[selected_features_rfe]
    
    # Strategy 2: Create polynomial features from the RFE-selected features
    from sklearn.preprocessing import PolynomialFeatures
    print("Generating polynomial features...")
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_rfe)
    X_test_poly = poly.transform(X_test_rfe)
    
    # Get feature names for polynomial features
    if hasattr(poly, 'get_feature_names_out'):
        poly_feature_names = poly.get_feature_names_out(selected_features_rfe)
    else:
        poly_feature_names = [f"poly_{i}" for i in range(X_poly.shape[1])]
    
    # Use time-based cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Train both on original and polynomial features
    lr_base = LogisticRegression(C=0.5, max_iter=10000, solver='liblinear', random_state=42)
    lr_poly = LogisticRegression(C=0.5, max_iter=10000, solver='liblinear', random_state=42)
    
    # Manual implementation of cross_val_predict for TimeSeriesSplit
    print("Performing time series cross-validation...")
    
    def manual_cross_val_predict(model, X, y, cv):
        """Manual implementation of cross_val_predict for TimeSeriesSplit"""
        y_pred = np.zeros(len(y))
        
        # Check if X is a DataFrame or numpy array
        is_dataframe = isinstance(X, pd.DataFrame)
        
        for train_idx, val_idx in cv.split(X, y):
            # Handle differently based on type
            if is_dataframe:
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train = y.iloc[train_idx]
            else:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
            
            # Train on this fold
            model.fit(X_train, y_train)
            
            # Predict on validation fold
            y_pred[val_idx] = model.predict_proba(X_val)[:,1]
        
        return y_pred
    
    # Cross-validation with AUC scoring
    y_pred_base = manual_cross_val_predict(lr_base, X_rfe, y, tscv)
    y_pred_poly = manual_cross_val_predict(lr_poly, X_poly, y, tscv)
    
    auc_base = roc_auc_score(y, y_pred_base)
    auc_poly = roc_auc_score(y, y_pred_poly)
    
    print(f"Base features AUC: {auc_base:.6f}")
    print(f"Polynomial features AUC: {auc_poly:.6f}")
    
    # Strategy 4: Use PCA on the full feature set
    print("Performing PCA...")
    pca = PCA(n_components=min(20, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Use the PCA features
    lr_pca = LogisticRegression(C=0.5, max_iter=10000, solver='liblinear', random_state=42)
    y_pred_pca = manual_cross_val_predict(lr_pca, pd.DataFrame(X_pca), y, tscv)
    auc_pca = roc_auc_score(y, y_pred_pca)
    print(f"PCA features AUC: {auc_pca:.6f}")
    
    # Strategy 5: Also try L1 regularization for built-in feature selection
    lr_l1 = LogisticRegression(penalty='l1', C=0.5, solver='liblinear', max_iter=10000, random_state=42)
    y_pred_l1 = manual_cross_val_predict(lr_l1, X_scaled, y, tscv)
    auc_l1 = roc_auc_score(y, y_pred_l1)
    print(f"L1 regularized AUC: {auc_l1:.6f}")
    
    # Strategy 6: Train XGBoost with carefully limited depth
    print("Training XGBoost model...")
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Manual cross-validation for XGBoost since cross_val_predict has compatibility issues
    xgb_preds = np.zeros(len(y))
    for train_idx, val_idx in tscv.split(X_scaled, y):
        X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train on this fold
        xgb.fit(X_train, y_train)
        
        # Predict on validation fold
        xgb_preds[val_idx] = xgb.predict_proba(X_val)[:,1]
    
    auc_xgb = roc_auc_score(y, xgb_preds)
    print(f"XGBoost AUC: {auc_xgb:.6f}")
    
    # Choose the best approach based on validation
    aucs = {
        'base': (auc_base, X_rfe, X_test_rfe, lr_base),
        'poly': (auc_poly, X_poly, X_test_poly, lr_poly),
        'pca': (auc_pca, X_pca, X_test_pca, lr_pca),
        'l1': (auc_l1, X_scaled, X_test_scaled, lr_l1),
        'xgb': (auc_xgb, X_scaled, X_test_scaled, xgb)
    }
    
    best_approach = max(aucs.items(), key=lambda x: x[1][0])
    best_name = best_approach[0]
    best_auc, X_final, X_test_final, best_model = best_approach[1]
    
    print(f"\nBest approach: {best_name} with AUC {best_auc:.6f}")
    
    # Strategy 7: Create an ensemble of the top 3 models
    top_approaches = sorted(aucs.items(), key=lambda x: x[1][0], reverse=True)[:3]
    print(f"Creating ensemble of top 3 models: {[name for name, _ in top_approaches]}")
    
    # Train final models
    final_models = []
    for name, (auc, X_train_data, _, model) in top_approaches:
        model.fit(X_train_data, y)
        final_models.append((name, model))
    
    # Strategy 8: Weight ensemble predictions by their validation AUC
    weights = [auc for _, (auc, _, _, _) in top_approaches]
    weights = np.array(weights) / sum(weights)  # Normalize weights
    
    # Generate predictions from each model
    ensemble_preds = np.zeros(len(test))
    for i, (name, (auc, X_train_data, X_test_data, model)) in enumerate(top_approaches):
        # Fit the model directly without calibration for now (which requires more fixes)
        model.fit(X_train_data, y)
        model_preds = model.predict_proba(X_test_data)[:,1]
        ensemble_preds += weights[i] * model_preds
        print(f"Added {name} predictions with weight {weights[i]:.4f}")
    
    # Create submission
    submission = pd.DataFrame({
        'id': test['id'],
        'rainfall': ensemble_preds
    })
    
    output_path = f"data/candidates/enhanced_ensemble_{best_auc:.6f}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    # Print feature importance from best model if available
    if hasattr(best_model, 'coef_') and best_name in ['base', 'l1']:
        if best_name == 'base':
            features = selected_features_rfe
        else:
            features = X_scaled.columns
            
        importance = pd.DataFrame({
            'feature': features,
            'coefficient': best_model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print("\nTop 10 most influential features:")
        print(importance.head(10).to_string(index=False))
        
        # Save to file
        importance.to_csv("data/candidates/enhanced_feature_importance.csv", index=False)
    elif best_name == 'xgb':
        importance = pd.DataFrame({
            'feature': X_scaled.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 most important features (XGBoost):")
        print(importance.head(10).to_string(index=False))
        
        # Save to file
        importance.to_csv("data/candidates/enhanced_feature_importance.csv", index=False)

if __name__ == "__main__":
    create_enhanced_logistic_submission()