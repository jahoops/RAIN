# filepath: /home/USER/Projects/RAIN/calibrated_validation.py
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

def extract_features(df):
    """Extract statistical features from submission"""
    rainfall = df['rainfall']
    features = {
        'mean': rainfall.mean(),
        'median': rainfall.median(),
        'std': rainfall.std(),
        'min': rainfall.min(),
        'max': rainfall.max(),
        'range': rainfall.max() - rainfall.min(),
        'skew': rainfall.skew(),
        'kurtosis': rainfall.kurtosis(),
        'unique_ratio': len(rainfall.unique()) / len(rainfall),
        'zero_count': (rainfall == 0).sum() / len(rainfall),
        'one_count': (rainfall == 1).sum() / len(rainfall),
        'out_of_range': ((rainfall < 0) | (rainfall > 1)).sum() / len(rainfall),
        'below_0.1': (rainfall < 0.1).sum() / len(rainfall),
        'above_0.9': (rainfall > 0.9).sum() / len(rainfall),
        'mid_range': ((rainfall >= 0.4) & (rainfall <= 0.6)).sum() / len(rainfall),
        'entropy': -np.sum(np.histogram(rainfall, bins=20, density=True)[0] * np.log2(np.histogram(rainfall, bins=20, density=True)[0] + 1e-10)) / 4.32,
    }
    return features

def validate_calibrated(submission_path, quiet=False):
    """Predict the actual competition score for a submission using a calibrated model"""
    if not quiet:
        print(f"Analyzing submission: {submission_path}")
    
    if not os.path.exists(submission_path):
        print(f"Error: File not found: {submission_path}")
        return None
    
    # Create model directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Check if score predictor exists
    score_predictor_path = 'models/score_predictor.pkl'
    if not os.path.exists(score_predictor_path):
        print("Score predictor model not found. Please run calibrate_validation.py first.")
        return None
    
    try:
        # Load score predictor
        with open(score_predictor_path, 'rb') as f:
            score_predictor = pickle.load(f)
        
        # Load submission
        submission_df = pd.read_csv(submission_path)
        
        # Check the format
        if 'id' not in submission_df.columns or 'rainfall' not in submission_df.columns:
            print("Error: Submission must have 'id' and 'rainfall' columns")
            return None
        
        # Extract features
        features = extract_features(submission_df)
        
        # Create feature vector in the right order
        feature_vector = [features[f] for f in score_predictor['features']]
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Predict score
        predicted_score = score_predictor['model'].predict(feature_vector)[0]
        
        # Ensure the score is in a reasonable range
        predicted_score = max(0.7, min(0.95, predicted_score))
        
        # Create output directory
        os.makedirs('analysis', exist_ok=True)
        
        # Create distribution visualization
        plt.figure(figsize=(10, 6))
        plt.hist(submission_df['rainfall'], bins=30, alpha=0.7)
        plt.title(f'Submission Distribution (Predicted Score: {predicted_score:.6f})')
        plt.xlabel('Rainfall')
        plt.ylabel('Count')
        plt.savefig('analysis/submission_distribution.png')
        
        if not quiet:
            print("\n=== CALIBRATED SCORE PREDICTION ===")
            
            # Print key statistics and their influence
            print("\nSubmission statistics:")
            for feature in ['mean', 'std', 'min', 'max', 'unique_ratio', 'out_of_range']:
                if feature in features:
                    print(f"{feature}: {features[feature]:.4f}")
            
            print(f"\nPredicted competition score: {predicted_score:.6f}")
            
            # Performance tier
            if predicted_score >= 0.9:
                print("Performance tier: OUTSTANDING")
            elif predicted_score >= 0.85:
                print("Performance tier: EXCELLENT")
            elif predicted_score >= 0.8:
                print("Performance tier: VERY GOOD")
            elif predicted_score >= 0.75:
                print("Performance tier: GOOD")
            else:
                print("Performance tier: NEEDS IMPROVEMENT")
            
            print("\nDistribution plot saved to analysis/submission_distribution.png")
        
        return {
            'predicted_score': predicted_score,
            'features': features
        }
        
    except Exception as e:
        print(f"Error validating submission: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import sys
    
    # Use first argument as submission path, or default to final submission
    submission_path = sys.argv[1] if len(sys.argv) > 1 else 'data/final/submission_final.csv'
    
    result = validate_calibrated(submission_path)
    
    if result:
        print(f"\nFinal predicted score: {result['predicted_score']:.6f}")
