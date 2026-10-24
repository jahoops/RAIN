import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from typing import List, Optional

def plot_feature_importance(model, feature_names: List[str], top_n: int = 20, 
                           figsize: tuple = (10, 8)) -> None:
    """
    Plot feature importance for a tree-based model
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to display
        figsize: Figure size
    """
    # Extract feature importances
    importances = model.feature_importances_
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Select top N features
    importance_df = importance_df.head(top_n)
    
    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray, 
                  figsize: tuple = (8, 6)) -> float:
    """
    Plot ROC curve and return AUC score
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        figsize: Figure size
    
    Returns:
        AUC score
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    
    return auc

def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Evaluate model performance using AUC
    
    Args:
        model: Trained model
        X: Feature data
        y: Target variable
    
    Returns:
        AUC score
    """
    predictions = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, predictions)
    print(f"AUC Score: {auc:.4f}")
    return auc
