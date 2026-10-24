import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel
import os
from optimized_binary_model import create_binary_optimized_features
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def create_optimized_logistic_submission():
    """Create a submission using optimized logistic regression trained only on first 146 samples"""
    # Load data
    train = pd.read_csv('/home/USER/Projects/RAIN/data/train.csv')
    test = pd.read_csv('/home/USER/Projects/RAIN/data/test.csv')

    # Generate features
    X, X_test = create_binary_optimized_features(train, test)
    y = train['rainfall']
    
    # Only use first 146 samples for training
    X_train_subset = X.iloc[:146]
    y_train_subset = y.iloc[:146]
    
    print(f"Training on first 146 samples only (out of {len(X)} total samples)")

    # No SMOTE needed as we're using a specific sample subset

    preprocessing = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Scale features - using only the training subset for fitting
    column_names = X.columns
    X_train_scaled = preprocessing.fit_transform(X_train_subset)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=column_names)
    
    # Transform the test set using parameters from training subset
    X_test_scaled = pd.DataFrame(preprocessing.transform(X_test), columns=column_names)
    
    print(f"Training optimized logistic regression with {X_train_scaled.shape[1]} features")
    
    # Create logistic regression with higher max_iter to solve convergence
    lr = LogisticRegression(
        C=1.0,  
        max_iter=10000,
        solver='liblinear',
        random_state=42
    )
    
    # Cross-validation with AUC scoring - only on the subset
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_proba = cross_val_predict(lr, X_train_scaled, y_train_subset, cv=kf, method='predict_proba')[:,1]
    auc_score = roc_auc_score(y_train_subset, y_pred_proba)
    print(f"Cross-validation AUC (146 samples only): {auc_score:.6f}")
    
    # Optional feature selection - can improve logistic regression
    selector = SelectFromModel(lr, threshold='mean')
    selector.fit(X_train_scaled, y_train_subset)
    selected_features = X_train_scaled.columns[selector.get_support()]
    
    if len(selected_features) < X_train_scaled.shape[1]:
        print(f"Selected {len(selected_features)} out of {X_train_scaled.shape[1]} features")
        X_selected = X_train_scaled[selected_features]
        X_test_selected = X_test_scaled[selected_features]
        
        # Retrain with selected features
        lr_selected = LogisticRegression(C=1.0, max_iter=10000, solver='liblinear', random_state=42)
        y_pred_proba_selected = cross_val_predict(
            lr_selected, X_selected, y_train_subset, cv=kf, method='predict_proba'
        )[:,1]
        auc_score_selected = roc_auc_score(y_train_subset, y_pred_proba_selected)
        print(f"Feature-selected AUC (146 samples only): {auc_score_selected:.6f}")
        
        # Use the better approach
        if auc_score_selected > auc_score:
            print("Using feature-selected model for final submission")
            auc_score = auc_score_selected
            X_final = X_selected
            X_test_final = X_test_selected
            lr_final = lr_selected
        else:
            print("Using full-feature model for final submission")
            X_final = X_train_scaled
            X_test_final = X_test_scaled
            lr_final = lr
    else:
        X_final = X_train_scaled
        X_test_final = X_test_scaled
        lr_final = lr
    
    # Hyperparameter tuning with GridSearchCV
    param_grid = {'C': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=10000, solver='liblinear', random_state=42),
        param_grid,
        cv=5,
        scoring='roc_auc'
    )
    grid_search.fit(X_final, y_train_subset)
    print(f"Best C value: {grid_search.best_params_['C']}")
    
    # Use the optimized C value
    lr_final = LogisticRegression(
        C=grid_search.best_params_['C'],
        max_iter=10000,
        solver='liblinear',
        random_state=42
    )
    
    # Train final model
    print("Training final model...")
    lr_final.fit(X_final, y_train_subset)
    
    # Train 5 models with different seeds
    seeds = [42, 123, 456, 789, 999]
    models = []
    for seed in seeds:
        model = LogisticRegression(C=1.0, max_iter=10000, solver='liblinear', random_state=seed)
        model.fit(X_final, y_train_subset)
        models.append(model)

    # Average predictions
    ensemble_preds = np.zeros(len(X_test_final))
    for model in models:
        ensemble_preds += model.predict_proba(X_test_final)[:,1]
    ensemble_preds /= len(models)
    
    # Generate predictions
    test_proba = ensemble_preds
    
    # Use same features but with XGBoost
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb.fit(X_final, y_train_subset)
    xgb_preds = xgb.predict_proba(X_test_final)[:,1]
    
    # Create submission
    submission = pd.DataFrame({
        'id': test['id'],
        'rainfall': test_proba
    })
    
    output_path = f"data/candidates/optimized_lr_smote_{auc_score:.6f}_2.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

    if hasattr(lr_final, 'coef_'):
        # For feature-selected model
        if 'selected_features' in locals():
            importance = pd.DataFrame({
                'feature': selected_features,
                'coefficient': lr_final.coef_[0]
            }).sort_values('coefficient', key=abs, ascending=False)
        else:
            importance = pd.DataFrame({
                'feature': X_train_scaled.columns,
                'coefficient': lr_final.coef_[0]
            }).sort_values('coefficient', key=abs, ascending=False)
            
        print("\nTop 10 most influential features:")
        print(importance.head(10).to_string(index=False))
        
        # Save to file
        importance.to_csv("data/candidates/lr_feature_importance.csv", index=False)

if __name__ == "__main__":
    create_optimized_logistic_submission()