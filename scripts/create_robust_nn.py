import tensorflow as tf
from tensorflow.keras import layers, regularizers
import pandas as pd
import numpy as np
import os
from data_pipeline import create_complete_data_pipeline

print("Loading data...")
X_train_temporal, X_test_temporal = create_complete_data_pipeline()
y_train = X_train_temporal['rainfall']

# Process features
X_train_features = X_train_temporal.drop(['id', 'rainfall'], axis=1)
X_test_features = X_test_temporal.drop(['id'], axis=1)

# Apply scaling
from sklearn.preprocessing import RobustScaler  # Use RobustScaler instead of StandardScaler
scaler = RobustScaler()  # More robust to outliers
X_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)

# Create a robust architecture to prevent overfitting
def create_distribution_robust_nn(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    
    # Gaussian noise layer to simulate distribution shifts
    x = layers.GaussianNoise(0.1)(inputs)
    
    # Simpler architecture with strong regularization
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.PReLU()(x)  # Learned activation function
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.PReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create model
model = create_distribution_robust_nn(X_scaled.shape[1])

# Compile with a robust loss function
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss='huber',  # Robust to outliers
    metrics=['mae']
)

# Train with careful validation
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6)
]

# Use a larger validation split to better detect overfitting
history = model.fit(
    X_scaled, y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.25,  # Larger validation split
    callbacks=callbacks,
    verbose=1
)

# Generate predictions
predictions = model.predict(X_test_scaled).flatten()

# Save predictions
submission = pd.DataFrame({
    'id': X_test_temporal['id'],
    'rainfall': predictions
})
submission.to_csv('/home/USER/Projects/RAIN/data/candidates/submission_robust_nn.csv', index=False)
print("Created robust neural network submission")