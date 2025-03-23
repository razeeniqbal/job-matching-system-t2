import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import os
import json
import shap
from .feature_engineering import FeatureEngineeringPipeline, get_raw_match_scores
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class XGBoostJobMatcher:
    """
    XGBoost model for job matching that predicts match scores based on
    job requirements and talent profiles.
    """
    
    def __init__(self, 
                 use_semantic_match=True,
                 xgb_params=None):
        # Initialize feature engineering pipeline
        self.feature_pipeline = FeatureEngineeringPipeline(
            use_semantic_match=use_semantic_match
        )
        
        # Default XGBoost parameters
        self.xgb_params = xgb_params or {
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
        
        # Initialize model
        self.model = xgb.XGBRegressor(**self.xgb_params)
        
        # For explainability
        self.explainer = None
        self.feature_names = None
    
    def prepare_data(self, data, generate_target=True):
        """
        Prepare data for training or prediction.
        If generate_target is True, calculate target scores for training.
        """
        # Extract features
        X_features = self.feature_pipeline.transform(data)
        
        # For training, generate target values
        if generate_target:
            # Calculate raw match scores based on weighted criteria
            raw_scores = get_raw_match_scores(data)
            y = np.array([item['score'] for item in raw_scores]) / 100.0  # Scale to 0-1 range
            return X_features, y
        
        return X_features
    
    def train(self, training_data, test_size=0.2, validation=True, save_model=True):
        """
        Train the XGBoost model using the provided training data.
        """
        logger.info("Fitting feature pipeline...")
        self.feature_pipeline.fit(training_data)
        
        logger.info("Preparing training data...")
        X_features, y = self.prepare_data(training_data)
        
        # Save feature names for explainability
        if hasattr(self.feature_pipeline, 'get_feature_names'):
            self.feature_names = self.feature_pipeline.get_feature_names()
        else:
            # If get_feature_names is not available, create generic names
            self.feature_names = [f'feature_{i}' for i in range(X_features.shape[1])]
        
        # Train with validation if requested
        if validation:
            logger.info("Splitting data for validation...")
            X_train, X_val, y_train, y_val = train_test_split(
                X_features, y, test_size=test_size, random_state=42
            )
            
            logger.info("Training model with validation...")
            
            # Simple approach without early stopping for compatibility with all XGBoost versions
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=True
            )
            
            # Evaluate on validation set
            val_preds = self.model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, val_preds))
            mae = mean_absolute_error(y_val, val_preds)
            r2 = r2_score(y_val, val_preds)
            
            logger.info(f"Validation RMSE: {rmse:.4f}")
            logger.info(f"Validation MAE: {mae:.4f}")
            logger.info(f"Validation R²: {r2:.4f}")
            
        else:
            logger.info("Training model on all data...")
            self.model.fit(X_features, y)
        
        # Initialize SHAP explainer
        logger.info("Creating SHAP explainer...")
        self.explainer = shap.TreeExplainer(self.model)
        
        # Save model if requested
        if save_model:
            self.save()
            
        return self
    
    def predict(self, data):
        """
        Predict match scores for new job applications.
        """
        # Prepare features
        X_features = self.prepare_data(data, generate_target=False)
        
        # Make predictions and scale to 0-100
        predictions = self.model.predict(X_features) * 100
        
        # Format results
        results = []
        for i, item in enumerate(data):
            job_app_id = item.get('job_app_id')
            score = round(float(predictions[i]), 2)
            results.append({
                'job_app_id': job_app_id,
                'score': score
            })
        
        return results
    
    def explain(self, data, plot=False):
        """
        Explain predictions using SHAP values.
        """
        if self.explainer is None:
            raise ValueError("Model must be trained before explanations can be generated")
        
        # Prepare features
        X_features = self.prepare_data(data, generate_target=False)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_features)
        
        # Create explanations
        explanations = []
        for i, item in enumerate(data):
            job_app_id = item.get('job_app_id')
            
            # Get features with their SHAP values
            feature_impacts = []
            for j, feature_name in enumerate(self.feature_names):
                impact = {
                    'feature': feature_name,
                    'impact': float(shap_values[i, j]),
                    'abs_impact': abs(float(shap_values[i, j]))
                }
                feature_impacts.append(impact)
            
            # Sort by absolute impact
            feature_impacts.sort(key=lambda x: x['abs_impact'], reverse=True)
            
            explanations.append({
                'job_app_id': job_app_id,
                'features': feature_impacts
            })
        
        # Plot summary if requested
        if plot:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_features, feature_names=self.feature_names)
            plt.tight_layout()
            
        return explanations
    
    def save(self, model_path='models/xgboost_model.joblib', pipeline_path='models/feature_pipeline.joblib'):
        """
        Save the model and feature pipeline to disk.
        """
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save the feature pipeline
        self.feature_pipeline.save(pipeline_path)
        logger.info(f"Feature pipeline saved to {pipeline_path}")
        
        # Save feature names
        feature_names_path = os.path.join(os.path.dirname(model_path), 'feature_names.json')
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_names, f)
        logger.info(f"Feature names saved to {feature_names_path}")
    
    @classmethod
    def load(cls, model_path='models/xgboost_model.joblib', pipeline_path='models/feature_pipeline.joblib'):
        """
        Load the model and feature pipeline from disk.
        """
        # Create a new instance
        instance = cls(use_semantic_match=False)  # Will be overridden by loaded pipeline
        
        # Load the feature pipeline
        instance.feature_pipeline = FeatureEngineeringPipeline.load(pipeline_path)
        logger.info(f"Feature pipeline loaded from {pipeline_path}")
        
        # Load the model
        instance.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load feature names if available
        feature_names_path = os.path.join(os.path.dirname(model_path), 'feature_names.json')
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                instance.feature_names = json.load(f)
            logger.info(f"Feature names loaded from {feature_names_path}")
        
        # Initialize SHAP explainer
        instance.explainer = shap.TreeExplainer(instance.model)
        
        return instance
    
    def evaluate(self, test_data):
        """
        Evaluate the model on test data.
        """
        # Prepare features and targets
        X_features, y_true = self.prepare_data(test_data)
        
        # Make predictions
        y_pred = self.model.predict(X_features)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Scale to 0-100 for display
        y_true_100 = y_true * 100
        y_pred_100 = y_pred * 100
        
        # Create a results DataFrame
        results_df = pd.DataFrame({
            'job_app_id': [item.get('job_app_id') for item in test_data],
            'true_score': y_true_100,
            'predicted_score': y_pred_100,
            'error': y_pred_100 - y_true_100
        })
        
        logger.info(f"Test RMSE: {rmse:.4f}")
        logger.info(f"Test MAE: {mae:.4f}")
        logger.info(f"Test R²: {r2:.4f}")
        
        return {
            'metrics': {
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            },
            'results': results_df
        }