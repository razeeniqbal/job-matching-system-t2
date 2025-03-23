# ui/model_info_tab.py - Enhanced with more explanations

import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from models.xgboost_job_matcher import XGBoostJobMatcher
from ui.components import display_styled_header

def load_model():
    """Load the trained XGBoost model if available, otherwise return None."""
    try:
        model = XGBoostJobMatcher.load()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def render_model_info_tab():
    """Render the model information tab UI with detailed explanations."""
    display_styled_header("Model Information")
    
    # Check if model exists
    model_path = "models/xgboost_model.joblib"
    
    if os.path.exists(model_path):
        # Model exists, show info
        model = load_model()
        
        if model:
            st.success("✅ XGBoost model is loaded and ready for predictions")
            
            # Model overview explanation
            st.subheader("Model Overview")
            st.markdown("""
            This system uses an **XGBoost (eXtreme Gradient Boosting)** model to predict job match scores. XGBoost is an advanced implementation of gradient boosted decision trees designed for speed and performance.
            
            ### How It Works
            The model evaluates multiple factors across four key dimensions:
            
            1. **Skills matching** - Compares required job skills with candidate skills using both direct matching and semantic similarity
            2. **Experience evaluation** - Analyzes years of experience relative to requirements
            3. **Education assessment** - Compares education levels using a hierarchical approach
            4. **Certification verification** - Evaluates required certifications against candidate's certifications
            
            The model combines these factors with learned weights to produce a final match score between 0-100.
            """)
            
            # Try to load feature importance
            try:
                # Get feature importance
                feature_importance = pd.DataFrame({
                    'Feature': [str(feature) for feature in model.feature_names],
                    'Importance': [float(imp) for imp in model.model.feature_importances_]
                })
                
                # Create feature name mapping for better readability
                feature_name_mapping = {
                    'skill_coverage': 'Skills Match Coverage',
                    'semantic_similarity': 'Skills Semantic Similarity',
                    'matching_skills_count': 'Number of Matching Skills',
                    'additional_skills': 'Additional Skills',
                    'total_skills': 'Total Skills Count',
                    'experience_ratio': 'Experience Level Ratio',
                    'experience_difference': 'Years Beyond Required',
                    'meets_experience_requirement': 'Meets Experience Requirement',
                    'education_difference': 'Education Level Difference',
                    'meets_education_requirement': 'Meets Education Requirement',
                    'exceeds_education_by': 'Education Levels Above Required',
                    'certification_coverage': 'Certification Coverage',
                    'matching_certifications': 'Matching Certifications',
                    'additional_certifications': 'Additional Certifications',
                    'has_all_required_certifications': 'Has All Required Certifications'
                }
                
                # Apply mapping
                feature_importance['Display Name'] = feature_importance['Feature'].map(
                    lambda x: feature_name_mapping.get(x, x)
                )
                
                feature_importance = feature_importance.sort_values('Importance', ascending=False)
                
                # Display feature importance with enhanced visualization
                st.subheader("Feature Importance")
                st.markdown("""
                The chart below shows which features have the greatest impact on match predictions. 
                Longer bars indicate greater influence on the final score.
                """)
                
                # Create interactive Plotly bar chart
                fig = px.bar(
                    feature_importance.head(15),
                    x='Importance',
                    y='Display Name',
                    orientation='h',
                    title='Top 15 Features by Importance',
                    color='Importance',
                    color_continuous_scale=px.colors.sequential.Blues
                )
                
                fig.update_layout(
                    xaxis_title="Relative Importance",
                    yaxis_title="Feature",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature explanations
                st.markdown("""
                ### Understanding Key Features
                
                #### Skills Features
                - **Skills Match Coverage**: Percentage of required skills that match the candidate's skills
                - **Skills Semantic Similarity**: How closely related the candidate's skills are to the job requirements
                - **Number of Matching Skills**: Raw count of directly matching skills
                - **Additional Skills**: Number of skills the candidate has beyond those required
                
                #### Experience Features
                - **Experience Level Ratio**: Candidate's experience divided by required experience
                - **Years Beyond Required**: Additional years of experience beyond the requirement
                - **Meets Experience Requirement**: Binary feature indicating if minimum experience is met
                
                #### Education Features
                - **Education Level Difference**: Gap between candidate's education and required level
                - **Meets Education Requirement**: Whether candidate meets or exceeds the education requirement
                - **Education Levels Above Required**: How many levels above the requirement the candidate is
                
                #### Certification Features
                - **Certification Coverage**: Percentage of required certifications that match
                - **Matching Certifications**: Number of matching certifications
                - **Has All Required Certifications**: Whether all required certifications are present
                """)
                
                # Model parameters
                st.subheader("Model Parameters")
                st.markdown("""
                These are the hyperparameters used to train the XGBoost model. Each parameter influences how the model learns from data:
                
                - **objective**: Defines the loss function to be optimized
                - **eval_metric**: Metric used to evaluate model performance during training
                - **learning_rate**: Controls how quickly the model adapts to the problem (smaller values = more conservative)
                - **max_depth**: Maximum depth of each tree (higher = more complex model)
                - **min_child_weight**: Minimum sum of instance weights needed in a child node
                - **subsample**: Fraction of samples used for training each tree (helps prevent overfitting)
                - **colsample_bytree**: Fraction of features used for building each tree
                - **n_estimators**: Number of trees in the ensemble
                """)
                
                params_df = pd.DataFrame(
                    [(str(k), str(v)) for k, v in model.xgb_params.items()], 
                    columns=['Parameter', 'Value']
                )
                st.dataframe(params_df, hide_index=True)
                
                # Model architecture visualization
                st.subheader("Model Architecture")
                st.markdown("""
                The diagram below illustrates the data flow through the job matching system.
                
                1. **Input**: Job requirements and talent profiles are collected
                2. **Feature Processing**: Raw data is transformed into features the model can understand
                3. **XGBoost Trees**: The model makes predictions based on learned patterns
                4. **Score Calculation**: Raw predictions are converted to a 0-100 score
                5. **SHAP Explanation**: The system generates explanations for why a score was given
                6. **Output**: Final score with explanations is presented to the user
                """)
                
                # Create a more visual representation of model architecture
                arch_fig = go.Figure()
                
                # Add model layers
                arch_fig.add_trace(go.Scatter(
                    x=[0, 1, 2, 3, 4, 5],
                    y=[0, 0, 0, 0, 0, 0],
                    mode="markers+text",
                    marker=dict(size=[30, 40, 50, 50, 40, 30], color=['#E3F2FD', '#90CAF9', '#42A5F5', '#1E88E5', '#1565C0', '#0D47A1']),
                    text=["Input", "Feature<br>Processing", "XGBoost<br>Trees", "Score<br>Calculation", "SHAP<br>Explanation", "Output"],
                    textposition="bottom center"
                ))
                
                # Add connecting lines
                for i in range(5):
                    arch_fig.add_shape(
                        type="line",
                        x0=i, y0=0,
                        x1=i+1, y1=0,
                        line=dict(color="#1976D2", width=3)
                    )
                
                arch_fig.update_layout(
                    title="Model Pipeline Flow",
                    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                    showlegend=False,
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=50)
                )
                
                st.plotly_chart(arch_fig, use_container_width=True)
                
                # Add model advantages section
                st.subheader("Advantages Over Rule-Based Systems")
                st.markdown("""
                This ML-based approach offers several advantages over traditional rule-based scoring:
                
                1. **Adaptability**: The model can learn from historical matching data to improve over time
                2. **Nuanced Understanding**: Goes beyond exact matching to understand semantic relationships between skills
                3. **Explainability**: Provides clear explanations for why a particular score was given
                4. **Customization**: Can be fine-tuned for different job categories or industries
                5. **Data-Driven Weights**: Learns which factors are truly important rather than using fixed weights
                """)
                
            except Exception as e:
                st.error(f"Error loading model details: {e}")
            
        else:
            st.error("Failed to load the XGBoost model")
    else:
        # Model doesn't exist
        st.warning(
            "⚠️ XGBoost model not found. Using rule-based scoring instead. "
            "Train a model using `python scripts/train_model.py` to enable ML-based matching."
        )
        
        st.subheader("How to Train a Model")
        st.markdown("""
        Follow these steps to train an XGBoost model for job matching:
        
        1. **Prepare your data**: Ensure you have job applications with requirements and talent profiles
        2. **Install dependencies**: Make sure all required packages are installed from requirements.txt
        3. **Run the training script**: Use one of the commands below
        """)
        
        st.code("""
# Train a model with default settings
python scripts/train_model.py

# Or customize training parameters
python scripts/train_model.py --data data/sample_jobs.json --test-size 0.2
        """)
        
        st.markdown("""
        ### Benefits of Training a Model
        
        Training an ML model will enable:
        - More accurate matching scores
        - Feature importance analysis
        - SHAP explanations for prediction transparency
        - Better handling of semantic relationships between skills
        """)