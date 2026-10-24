import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import xgboost as xgb
import argparse

def load_model(model_path):
    """Load the trained model from pickle file"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_feature_names(feature_names_path):
    """Load feature names"""
    if os.path.exists(feature_names_path):
        return pd.read_csv(feature_names_path).iloc[:, 0].tolist()
    return None

def plot_feature_importance(model, feature_names, output_path=None):
    """Plot feature importance"""
    if not hasattr(model, 'feature_importances_'):
        print("Model doesn't have feature_importances_ attribute")
        return
        
    # Get feature importance
    importance = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance Ranking')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Feature importance plot saved to {output_path}")
    else:
        plt.show()

def analyze_model_performance(model, X_val, y_val, feature_names=None):
    """Analyze model performance on validation data"""
    print("Analyzing model performance...")
    
    # Make predictions
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_val)[:, 1]
    else:
        y_prob = model.predict(X_val)
    
    y_pred = (y_prob > 0.5).astype(int)
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('model_roc_curve.png')
    print("ROC curve saved to model_roc_curve.png")
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Feature importance
    if feature_names:
        plot_feature_importance(model, feature_names, 'feature_importance.png')

def main(args):
    # Load model
    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"Model file does not exist: {model_path}")
        return
        
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    # Load feature names
    feature_names_path = args.feature_names_path
    feature_names = load_feature_names(feature_names_path)
    
    if feature_names:
        print(f"Loaded {len(feature_names)} feature names")
        
        # Print top 10 features by importance (if available)
        if hasattr(model, 'feature_importances_'):
            indices = np.argsort(model.feature_importances_)[::-1]
            print("\nTop 10 features by importance:")
            for i in range(min(10, len(feature_names))):
                print(f"{feature_names[indices[i]]}: {model.feature_importances_[indices[i]]:.4f}")
    
    # If validation data is provided, analyze performance
    if args.val_data and args.val_target:
        if not os.path.exists(args.val_data) or not os.path.exists(args.val_target):
            print(f"Validation data or target file does not exist")
        else:
            X_val = pd.read_csv(args.val_data)
            y_val = pd.read_csv(args.val_target).iloc[:, 0]
            analyze_model_performance(model, X_val, y_val, feature_names)
    else:
        # Just plot feature importance
        if feature_names:
            plot_feature_importance(model, feature_names, 'feature_importance.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze trained model performance')
    parser.add_argument('--model_path', type=str, default='models/xgboost_model.pkl',
                        help='Path to the trained model file')
    parser.add_argument('--feature_names_path', type=str, default='models/feature_names.csv',
                        help='Path to the feature names file')
    parser.add_argument('--val_data', type=str, help='Path to validation data (optional)')
    parser.add_argument('--val_target', type=str, help='Path to validation target (optional)')
    
    args = parser.parse_args()
    main(args)
