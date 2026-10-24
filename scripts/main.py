import argparse
import pandas as pd
from cross_validation import train_with_cv
from validation_with_synthetic import validate_with_synthetic_data

def main():
    parser = argparse.ArgumentParser(description='RAIN Project - Rainfall Prediction')
    parser.add_argument('--mode', type=str, default='all', choices=['cv', 'synthetic', 'all'],
                        help='Validation mode to run (cv, synthetic, or all)')
    parser.add_argument('--feature-strategy', type=str, default='all',
                        choices=['all', 'minimal', 'boruta_top', 'rainfall_patterns', 'all_with_patterns'],
                        help='Feature engineering strategy to use')
    parser.add_argument('--model', type=str, default='xgboost',
                        choices=['xgboost', 'lightgbm', 'catboost'],
                        help='Model to use for predictions')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of CV folds')
    args = parser.parse_args()
    
    results = {}
    
    if args.mode in ['cv', 'all']:
        print(f"\n{'-'*50}")
        print(f"Running cross-validation with {args.feature_strategy} features")
        print(f"{'-'*50}")
        cv_score = train_with_cv(n_folds=args.n_folds, feature_strategy=args.feature_strategy)
        results['CV Score'] = cv_score
    
    if args.mode in ['synthetic', 'all']:
        print(f"\n{'-'*50}")
        print(f"Running synthetic data validation with {args.feature_strategy} features")
        print(f"{'-'*50}")
        synthetic_score = validate_with_synthetic_data(model_name=args.model, 
                                                     feature_strategy=args.feature_strategy)
        results['Synthetic Score'] = synthetic_score
    
    # Print final results
    print("\nFinal Results:")
    for key, value in results.items():
        print(f"{key}: {value:.6f}")

if __name__ == "__main__":
    main()
