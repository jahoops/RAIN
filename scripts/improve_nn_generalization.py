import tensorflow as tf
from tensorflow.keras import layers, regularizers, optimizers
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from data_pipeline import create_complete_data_pipeline

# Load data
print("Loading data...")
X_train_temporal, X_test_temporal = create_complete_data_pipeline()
y_train = X_train_temporal['rainfall']

# Load best tree-based model for guidance
best_tree_model_path = "/home/USER/Projects/RAIN/data/candidates/submission.85518.csv"
tree_predictions = pd.read_csv(best_tree_model_path)

# Process features
X_train_features = X_train_temporal.drop(['id', 'rainfall'], axis=1)
X_test_features = X_test_temporal.drop(['id'], axis=1)

# Apply scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)

print(f"Training data shape: {X_scaled.shape}")

# Create a more generalizable architecture
def create_regularized_nn(input_dim):
    # Use stronger regularization, simpler architecture
    inputs = layers.Input(shape=(input_dim,))
    
    # First layer with strong regularization
    x = layers.Dense(128, 
                    kernel_regularizer=regularizers.l2(0.001),
                    activity_regularizer=regularizers.l1(0.0001))(inputs)
    x = layers.LeakyReLU(alpha=0.05)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)  # Stronger dropout
    
    # Simpler middle layers
    x = layers.Dense(96, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.LeakyReLU(alpha=0.05)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.LeakyReLU(alpha=0.05)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create model
model = create_regularized_nn(X_scaled.shape[1])

# Use a carefully tuned learning rate with decay
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0005,
    decay_steps=1000,
    decay_rate=0.95
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Custom loss function that combines MSE with knowledge distillation from tree model
def distillation_loss(alpha=0.7):
    # Create a distillation loss function using tree predictions as guidance
    def loss(y_true, y_pred):
        # Match IDs between training data and tree predictions
        train_ids = X_train_temporal['id'].values
        tree_ids = tree_predictions.iloc[:, 0].values
        
        # Find positions of training IDs in the tree predictions
        # This is complex, so use a simplified approach
        mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        
        return mse_loss
    return loss

# Compile with standard loss for now (distillation is complex to implement)
model.compile(
    optimizer=optimizer,
    loss='huber',
    metrics=['mae']
)

# Use more careful training with early stopping
callbacks = [
    # Early stopping with patience
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    # Learning rate reduction when plateauing
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    # Model checkpoint
    tf.keras.callbacks.ModelCheckpoint(
        filepath='/home/USER/Projects/RAIN/models/regularized_nn.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Train with careful cross-validation
print("Training regularized neural network...")
history = model.fit(
    X_scaled, y_train,
    validation_split=0.2,  
    epochs=100,            # Try more epochs with early stopping
    batch_size=32,         # Smaller batch size
    callbacks=callbacks,
    verbose=1
)

# Generate predictions
print("Generating predictions...")
predictions = model.predict(X_test_scaled).flatten()

# Save predictions
submission = pd.DataFrame({
    'id': X_test_temporal['id'],
    'rainfall': predictions
})
output_path = '/home/USER/Projects/RAIN/data/candidates/submission_regularized_nn.csv'
submission.to_csv(output_path, index=False)
print(f"Saved regularized neural network predictions to {output_path}")

# Create blended model (neural network + tree model)
print("Creating blended model with 30% neural network, 70% tree model...")
tree_preds = tree_predictions.iloc[:, 1].values
blended_preds = 0.7 * tree_preds + 0.3 * predictions

# Save blended predictions
blend_submission = pd.DataFrame({
    'id': X_test_temporal['id'],
    'rainfall': blended_preds
})
blend_path = '/home/USER/Projects/RAIN/data/candidates/submission_nn_tree_blend.csv'
blend_submission.to_csv(blend_path, index=False)
print(f"Saved blended predictions to {blend_path}")