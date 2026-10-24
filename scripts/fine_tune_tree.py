import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from data_pipeline import create_complete_data_pipeline

# Load data
X_train_temporal, X_test_temporal = create_complete_data_pipeline()
y_train = X_train_temporal['rainfall']

# Process features
X_train_features = X_train_temporal.drop(['id', 'rainfall'], axis=1)
X_test_features = X_test_temporal.drop(['id'], axis=1)

# Create a fine-tuned random forest
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=0.7,  # Slightly different feature ratio for diversity
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=43  # Different random seed for diversity
)

# Fit model
print("Training fine-tuned random forest...")
rf.fit(X_train_features, y_train)

# Generate predictions
predictions = rf.predict(X_test_features)

# Save predictions
submission = pd.DataFrame({
    'id': X_test_temporal['id'],
    'rainfall': predictions
})
submission.to_csv('/home/USER/Projects/RAIN/data/candidates/fine_tuned_rf.csv', index=False)
print("Created fine-tuned random forest submission")