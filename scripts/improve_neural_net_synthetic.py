import tensorflow as tf
from tensorflow.keras import layers, regularizers
import pandas as pd
import numpy as np
import os

# Replace the synthetic data loading section with:

print("Loading raw synthetic data...")
try:
    # Try to load raw synthetic data without engineered features
    synthetic_data = pd.read_csv('/home/USER/Projects/RAIN/data/raw_synthetic_data.csv')
    print(f"Loaded raw synthetic data with shape: {synthetic_data.shape}")
except FileNotFoundError:
    # If not found, generate synthetic data from the original training set
    print("Raw synthetic file not found - using original training data...")
    from data_loader import load_data
    synthetic_data, _ = load_data()  # Just use the original training data
    print(f"Using original training data with shape: {synthetic_data.shape}")

# Import required functions
from data_pipeline import preprocess_data, engineer_features, add_temporal_rainfall_features

# Step 1: Load raw test.csv data
print("1. Loading raw data...")
from data_loader import load_data
_, X_test_temporal = load_data()

# Step 2: Basic preprocessing
print("2. Applying basic preprocessing...")
synthetic_data, X_test_temporal = preprocess_data(synthetic_data, X_test_temporal)

# Step 3: Basic feature engineering
print("3. Creating basic engineered features...")
synthetic_data, X_test_temporal = engineer_features(synthetic_data, X_test_temporal)

# Step 4: Temporal features (after basic preprocessing and feature engineering)
print("4. Adding temporal rainfall features...")
synthetic_data, X_test_temporal = add_temporal_rainfall_features(synthetic_data, X_test_temporal)

# Extract target
y_train = synthetic_data['rainfall']

# Normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Extract features (exclude id and target)
X_train_features = synthetic_data.drop(['id', 'rainfall'], axis=1)
X_test_features = X_test_temporal.drop(['id'], axis=1)

# Apply scaling
X_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)

print(f"Training data shape after processing: {X_scaled.shape}")
print(f"Testing data shape after processing: {X_test_scaled.shape}")

# Create a network architecture that leverages temporal data
def create_temporal_nn(input_dim):
    # Use a more specialized architecture for temporal data
    inputs = layers.Input(shape=(input_dim,))
    
    # Initial processing
    x = layers.Dense(256)(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Rest of architecture remains the same
    for i in range(3):
        residual = x
        x = layers.Dense(256)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(256)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        
        # Add residual connection
        x = layers.Add()([x, residual])
    
    # Output layer - change to linear activation since rainfall is not bounded [0,1]
    outputs = layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create model with correct input dimension
model = create_temporal_nn(X_scaled.shape[1])

# Use a different optimizer than you might have used before
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=0.001,
    weight_decay=1e-4
)

# Compile with different loss function
model.compile(
    optimizer=optimizer,
    loss='huber',  # More robust to outliers than MSE
    metrics=['mae']
)

# Train with callbacks
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
    tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
]

model.fit(
    X_scaled, y_train,
    epochs=200,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Predict and save
predictions = model.predict(X_test_scaled).flatten()

# Save raw predictions for backup
np.save("/home/USER/Projects/RAIN/data/nn_synthetic_predictions.npy", predictions)

# Create submission directory if it doesn't exist
os.makedirs("/home/USER/Projects/RAIN/data/candidates", exist_ok=True)

# Create submission with proper IDs
submission = pd.DataFrame({
    'id': X_test_temporal['id'],
    'rainfall': predictions
})

# Save with robust error handling
try:
    output_path = '/home/USER/Projects/RAIN/data/candidates/submission_synthetic_nn.csv'
    submission.to_csv(output_path, index=False)
    print(f"Created neural network submission: {output_path}")
    print(f"File size: {os.path.getsize(output_path)} bytes")
except Exception as e:
    print(f"Error saving submission file: {e}")
    # Fallback save location
    submission.to_csv('/home/USER/Projects/RAIN/submission_synthetic_nn.csv', index=False)
    print("Saved to fallback location in project root directory")