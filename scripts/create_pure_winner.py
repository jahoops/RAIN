import pandas as pd

# Load the best performing model
best_model = pd.read_csv("/home/USER/Projects/RAIN/data/candidates/submission.85518.csv")

# Create a pure submission with very slight adjustments
# This helps avoid any potential model averaging that might reduce the score
pure_model = pd.DataFrame({
    'id': best_model.iloc[:, 0],
    'rainfall': best_model.iloc[:, 1] * 1.001  # Tiny adjustment that won't hurt
})

pure_model.to_csv('/home/USER/Projects/RAIN/data/candidates/pure_winner.csv', index=False)
print("Created pure winner submission")