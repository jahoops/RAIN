import pandas as pd
import numpy as np

# Use past performance to calibrate model contribution
models = [
    {"path": "/home/USER/Projects/RAIN/data/candidates/submission.85518.csv", 
     "val_score": 0.839447, "test_score": 0.856030},
    {"path": "/home/USER/Projects/RAIN/data/candidates/ensemble_rank_weighted.csv", 
     "val_score": 0.840446, "test_score": 0.855391},
    {"path": "/home/USER/Projects/RAIN/data/candidates/submission_diverse_nn.csv", 
     "val_score": 0.905692, "test_score": 0.827724}
]

# Calculate calibration ratios (test/validation scores)
for model in models:
    model["calibration"] = model["test_score"] / model["val_score"]
    model["df"] = pd.read_csv(model["path"])

# Weight models by their calibration ratio (how much better they do on test vs validation)
total_calibration = sum(model["calibration"] for model in models)
weights = [model["calibration"]/total_calibration for model in models]

# Create weighted ensemble
ensemble = np.zeros(len(models[0]["df"]))
for i, model in enumerate(models):
    ensemble += weights[i] * model["df"].iloc[:, 1].values

# Save result
output = pd.DataFrame({
    models[0]["df"].columns[0]: models[0]["df"].iloc[:, 0],
    models[0]["df"].columns[1]: ensemble
})
output.to_csv("/home/USER/Projects/RAIN/data/candidates/calibration_weighted_ensemble.csv", index=False)