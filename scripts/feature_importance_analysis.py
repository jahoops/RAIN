import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# Load models from each strategy
all_model = pickle.load(open('models/xgb_fold_0.pkl', 'rb'))
min_model_path = 'models/minimal_xgb_fold_0.pkl'

# Load feature names
all_features = pd.read_csv('models/feature_names.csv')['feature'].tolist()
min_features = ['cloud', 'humidity', 'windspeed', 'dewpoint', 'temparature']

# Plot feature importance comparison
plt.figure(figsize=(12, 8))
plt.barh(all_features, all_model.feature_importances_)
plt.xlabel('Feature Importance')
plt.title('Feature Importance - All Features')
plt.tight_layout()
plt.savefig('models/all_features_importance.png')
print("Feature importance plot saved to models/all_features_importance.png")
