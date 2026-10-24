import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import os

def build_model(input_dim):
    """Build a neural network model"""
    model = Sequential([
        Dense(256, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

def train_neural_model(n_folds=5):
    """Train a neural network model with cross-validation"""
    # Load data
    print("Loading data...")
    try:
        train_df = pd.read_csv('data/train_advanced_features.csv')
        test_df = pd.read_csv('data/test_advanced_features.csv')
    except:
        print("Advanced feature files not found. Please run advanced_features.py first")
        from data_loader import load_data, preprocess_data
        from advanced_features import create_advanced_features
        
        train_df, test_df = load_data()
        train_df, test_df = preprocess_data(train_df, test_df)
        train_df, test_df = create_advanced_features(train_df, test_df)
    
    # Prepare features and target
    X = train_df.drop(['id', 'rainfall'], axis=1)
    y = train_df['rainfall']
    X_test = test_df.drop(['id'], axis=1)
    
    # Ensure feature consistency
    common_cols = list(set(X.columns).intersection(set(X_test.columns)))
    X = X[common_cols]
    X_test = X_test[common_cols]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    # Set up cross-validation
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize arrays for predictions
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    
    # Directory for saving models
    os.makedirs('models/neural', exist_ok=True)
    
    # Train on folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\nTraining fold {fold+1}/{n_folds}")
        
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Define callbacks
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ]
        
        # Build and train model
        model = build_model(X_train.shape[1])
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=32,
            callbacks=callbacks,
            verbose=2
        )
        
        # Save model
        model.save(f'models/neural/nn_fold_{fold}.h5')
        
        # Make predictions
        oof_preds[val_idx] = model.predict(X_val).flatten()
        test_preds += model.predict(X_test_scaled).flatten() / n_folds
        
        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_val, oof_preds[val_idx])
        fold_scores.append(auc)
        print(f"Fold {fold+1} AUC: {auc:.6f}")
        
        # Plot learning curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title(f'Fold {fold+1} Loss')
        
        plt.subplot(1, 2, 2)
        # Find the correct AUC key in history dictionary
        auc_key = None
        val_auc_key = None
        for key in history.history.keys():
            if 'auc' in key.lower() and not key.startswith('val_'):
                auc_key = key
            if 'auc' in key.lower() and key.startswith('val_'):
                val_auc_key = key

        if auc_key and val_auc_key:
            plt.plot(history.history[auc_key], label='Train AUC')
            plt.plot(history.history[val_auc_key], label='Validation AUC')
        else:
            print(f"Warning: AUC metrics not found. Available keys: {history.history.keys()}")
        
        plt.legend()
        plt.title(f'Fold {fold+1} AUC')
        
        plt.savefig(f'models/neural/learning_curves_fold_{fold}.png')
    
    # Calculate overall AUC
    cv_auc = roc_auc_score(y, oof_preds)
    print(f"\nCV AUC: {cv_auc:.6f}")
    print(f"Fold AUCs: {[round(score, 6) for score in fold_scores]}")
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'rainfall': test_preds
    })
    
    # Save submission
    submission_path = 'data/submission_neural.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    
    # Copy to keep directory
    import shutil
    os.makedirs('data/keep', exist_ok=True)
    shutil.copy2(submission_path, 'data/keep/submission_neural.csv')
    
    # Save OOF predictions
    oof_df = pd.DataFrame({
        'id': train_df['id'],
        'true_rainfall': y,
        'predicted_rainfall': oof_preds
    })
    oof_df.to_csv('models/neural/oof_predictions.csv', index=False)
    
    # Validate submission
    try:
        from validate_submission import validate_submission
        result = validate_submission(submission_path)
        if result and 'estimated_score' in result:
            print(f"Estimated score: {result['estimated_score']:.6f}")
    except:
        print("Could not validate submission")
    
    return {
        'cv_auc': cv_auc,
        'fold_scores': fold_scores,
        'oof_preds': oof_preds,
        'test_preds': test_preds
    }

if __name__ == "__main__":
    print("Training neural network model...")
    results = train_neural_model()
    print("Neural network training complete!")
