#!/usr/bin/env python
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.xgboost_job_matcher import XGBoostJobMatcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    Load the dataset from a JSON file.
    """
    logger.info(f"Loading data from {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def train_and_evaluate(data_path, test_size=0.2, output_dir='models', plot=True):
    """
    Train the XGBoost job matcher model and evaluate its performance.
    """
    # Load the data
    job_applications = load_data(data_path)
    logger.info(f"Loaded {len(job_applications)} job applications")
    
    # Split data for training and testing
    train_data, test_data = train_test_split(
        job_applications, 
        test_size=test_size, 
        random_state=42
    )
    
    logger.info(f"Training set size: {len(train_data)}")
    logger.info(f"Test set size: {len(test_data)}")
    
    # Initialize and train the model
    model = XGBoostJobMatcher(
        use_semantic_match=True,
        xgb_params={
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 100,
            'random_state': 42
        }
    )
    
    logger.info("Training model...")
    model.train(
        train_data, 
        test_size=0.2,  # Further split training data for validation
        validation=True,
        save_model=False  # We'll save after evaluation
    )
    
    # Evaluate the model on the test set
    logger.info("Evaluating model on test data...")
    eval_results = model.evaluate(test_data)
    
    # Get predictions
    predictions = model.predict(test_data)
    
    # Display results
    logger.info("Model evaluation results:")
    for metric, value in eval_results['metrics'].items():
        logger.info(f"{metric.upper()}: {value:.4f}")
    
    # Generate explanations for test data
    logger.info("Generating SHAP explanations...")
    explanations = model.explain(test_data[:5], plot=plot)  # Only explain a few examples
    
    # Save the model
    model_path = os.path.join(output_dir, 'xgboost_model.joblib')
    pipeline_path = os.path.join(output_dir, 'feature_pipeline.joblib')
    model.save(model_path, pipeline_path)
    
    # Save evaluation results
    results_df = eval_results['results']
    results_path = os.path.join(output_dir, 'evaluation_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"Evaluation results saved to {results_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame(predictions)
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to {predictions_path}")
    
    # Generate and save performance visualizations if requested
    if plot:
        # Create output directory for plots if it doesn't exist
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Predicted vs Actual Scores
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['true_score'], results_df['predicted_score'], alpha=0.7)
        plt.plot([0, 100], [0, 100], 'r--')
        plt.xlabel('Actual Score')
        plt.ylabel('Predicted Score')
        plt.title('Predicted vs Actual Scores')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, 'pred_vs_actual.png'))
        
        # 2. Error Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(results_df['error'], bins=20, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, 'error_distribution.png'))
        
        logger.info(f"Performance plots saved to {plots_dir}")
    
    return model, eval_results

def main():
    parser = argparse.ArgumentParser(description='Train an XGBoost job matching model')
    
    parser.add_argument(
        '--data', 
        type=str, 
        default='data/sample_jobs.json',
        help='Path to the job applications dataset'
    )
    
    parser.add_argument(
        '--test-size', 
        type=float, 
        default=0.2,
        help='Proportion of data to use for testing'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='models',
        help='Directory to save model and results'
    )
    
    parser.add_argument(
        '--no-plot', 
        action='store_false',
        dest='plot',
        help='Do not generate performance plots'
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train and evaluate the model
    train_and_evaluate(
        data_path=args.data,
        test_size=args.test_size,
        output_dir=args.output_dir,
        plot=args.plot
    )

if __name__ == '__main__':
    main()