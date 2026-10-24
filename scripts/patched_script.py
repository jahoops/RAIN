"""
Script with pandas compatibility fixes applied
"""

# Apply the compatibility patch first, before any other imports
from utils.pandas_compat import fix_pandas_compatibility, patch_seaborn
restore_pandas = fix_pandas_compatibility()
patch_seaborn()

# Now proceed with normal imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # This should now work without errors

def run_visualization():
    """Run a visualization that uses seaborn"""
    print("Running visualization with patched seaborn...")
    
    # Load some example data - replace with your actual data loading code
    try:
        df = pd.read_csv('data/train_advanced_features.csv')
        print(f"Loaded data with {len(df)} rows")
    except Exception as e:
        print(f"Could not load data: {e}")
        # Create some dummy data instead
        df = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100)
        })
    
    # Create a figure
    plt.figure(figsize=(10, 8))
    
    # Try using seaborn with the patch applied
    try:
        # This is the type of call that would trigger the error
        if 'rainfall' in df.columns:
            sns.histplot(df['rainfall'], bins=30)
            plt.title("Rainfall Distribution (Patched)")
        else:
            sns.histplot(df['x'], bins=30)
            plt.title("X Distribution (Patched)")
            
        plt.tight_layout()
        plt.savefig('patched_visualization.png')
        print("Visualization saved to patched_visualization.png")
    except Exception as e:
        print(f"Error in visualization: {e}")
    
    return "Visualization complete"

if __name__ == "__main__":
    run_visualization()
    print("Done!")
