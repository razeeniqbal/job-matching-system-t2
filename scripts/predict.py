#!/usr/bin/env python
import os
import sys
import json
import argparse
import pandas as pd
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
    Load job applications from a JSON file.
    """
    logger.info(f"Loading data from {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def predict_scores(model_path, pipeline_path, data_path, output_path, explain=False):
    """
    Predict match scores using a trained model.
    """
    # Load the model
    logger.info(f"Loading model from {model_path}")
    model = XGBoostJobMatcher.load(model_path, pipeline_path)
    
    # Load the data
    job_applications = load_data(data_path)
    logger.info(f"Loaded {len(job_applications)} job applications")
    
    # Make predictions
    logger.info("Generating predictions...")
    predictions = model.predict(job_applications)
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    # Print predictions
    logger.info("Predictions:")
    for pred in predictions:
        logger.info(f"Job Application {pred['job_app_id']}: Score = {pred['score']:.2f}")
    
    # Generate explanations if requested
    if explain:
        logger.info("Generating explanations...")
        explanations = model.explain(job_applications)
        
        # Save explanations to JSON
        explain_path = os.path.splitext(output_path)[0] + '_explanations.json'
        with open(explain_path, 'w') as f:
            json.dump(explanations, f, indent=2)
        logger.info(f"Explanations saved to {explain_path}")
        
        # Print top factors for each prediction
        logger.info("Top factors affecting predictions:")
        for expl in explanations:
            job_app_id = expl['job_app_id']
            logger.info(f"Job Application {job_app_id}:")
            
            # Show top 3 factors
            for i, factor in enumerate(expl['features'][:3]):
                impact = factor['impact']
                feature = factor['feature']
                direction = "increased" if impact > 0 else "decreased"
                logger.info(f"  {i+1}. {feature} {direction} score by {abs(impact):.4f}")
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Predict job match scores using XGBoost model')
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='models/xgboost_model.joblib',
        help='Path to the trained model'
    )
    
    parser.add_argument(
        '--pipeline', 
        type=str, 
        default='models/feature_pipeline.joblib',
        help='Path to the feature pipeline'
    )
    
    parser.add_argument(
        '--data', 
        type=str, 
        default='data/sample_jobs.json',
        help='Path to the job applications data'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='predictions.csv',
        help='Path to save the predictions'
    )
    
    parser.add_argument(
        '--explain', 
        action='store_true',
        help='Generate explanations for predictions'
    )
    
    args = parser.parse_args()
    
    # Predict scores
    predict_scores(
        model_path=args.model,
        pipeline_path=args.pipeline,
        data_path=args.data,
        output_path=args.output,
        explain=args.explain
    )

if __name__ == '__main__':
    main()