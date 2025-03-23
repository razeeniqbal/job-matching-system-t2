# config/settings.py
# Constants and configuration parameters

# Education options
EDUCATION_OPTIONS = [
    "High School", 
    "Diploma", 
    "Associate", 
    "Bachelor's", 
    "Master's", 
    "PhD"
]

# Education hierarchy mapping
EDUCATION_HIERARCHY = {
    'High School': 1,
    'Diploma': 2,
    'Associate': 3,
    'Bachelor\'s': 4,
    'Master\'s': 5,
    'PhD': 6
}

# Score color scheme
SCORE_COLORS = {
    'excellent': '#4CAF50',  # Green
    'good': '#2196F3',       # Blue
    'partial': '#FFC107',    # Amber
    'poor': '#F44336'        # Red
}

# Default XGBoost parameters
DEFAULT_XGB_PARAMS = {
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

# Default model and data paths
MODEL_PATH = "models/xgboost_model.joblib"
PIPELINE_PATH = "models/feature_pipeline.joblib"
SAMPLE_DATA_PATH = "data/sample_jobs.json"