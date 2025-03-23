# ui/json_input_tab.py
import streamlit as st
import pandas as pd
import json
from models.xgboost_job_matcher import XGBoostJobMatcher
from models.feature_engineering import get_raw_match_scores
from utils.data_loader import load_data_from_json_string
from ui.visualizations import create_batch_visualization, create_score_gauge, visualize_feature_importance
from ui.components import display_styled_header, display_summary_metrics

def load_model():
    """Load the trained XGBoost model if available, otherwise return None."""
    try:
        model = XGBoostJobMatcher.load()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def render_json_input_tab():
    """Render the JSON input tab UI."""
    display_styled_header("Process JSON Input")
    
    st.write("Enter job applications in JSON format:")
    
    # Example JSON
    default_json = """
[ 
{ 
    "job_app_id": 201, 
    "job_requirements": { 
        "core_skills": ["Python", "Machine Learning", "SQL"], 
        "min_experience": 3,   
        "education": "Bachelor's",   
        "certifications": ["AWS Certified"] 
    }, 


    "talent_profile": { 
        "skills": ["Python", "Machine Learning", "Deep Learning", "SQL"], 
        "experience": 4,   
        "education": "Master's", 
        "certifications": ["AWS Certified", "Google Cloud Certified"] 
    } 
}, 
{ 
    "job_app_id": 202, 
    "job_requirements": { 
        "core_skills": ["Java", "Spring Boot", "Microservices"], 
        "min_experience": 5,   
        "education": "Bachelor's",   
        "certifications": ["Oracle Java Certified"] 
    }, 
    "talent_profile": { 
        "skills": ["Java", "Spring Boot", "Microservices", "Kafka"], 
        "experience": 3,   
        "education": "Diploma", 
        "certifications": ["Oracle Java Certified"] 
    } 
} 
]
"""
    
    json_input = st.text_area(
        "JSON Input",
        value=default_json,
        height=300
    )
    
    # Process button
    if st.button("Process JSON Input", key="process_json", type="primary"):
        try:
            # Parse JSON
            job_applications = load_data_from_json_string(json_input)
            
            if job_applications:
                st.success(f"Successfully parsed {len(job_applications)} job applications")
                
                # Calculate scores
                model = load_model()
                
                if model:
                    # Use ML model
                    results = model.predict(job_applications)
                    explanations = model.explain(job_applications)
                    using_ml = True
                else:
                    # Fallback to rule-based
                    results = get_raw_match_scores(job_applications)
                    explanations = None
                    using_ml = False
                
                # Display results
                display_styled_header("Matching Results")
                st.info(f"Using {'ML model' if using_ml else 'rule-based scoring'} for match calculation")
                
                # Create DataFrame for display
                results_df = pd.DataFrame(results)
                # Ensure proper data types
                results_df['job_app_id'] = results_df['job_app_id'].astype(str)
                results_df['score'] = results_df['score'].astype(float)
                results_df = results_df.sort_values('score', ascending=False)
                
                # Display results table
                st.dataframe(
                    results_df,
                    column_config={
                        'job_app_id': st.column_config.TextColumn('Job Application ID'),
                        'score': st.column_config.ProgressColumn('Match Score', format='%.1f%%', min_value=0, max_value=100)
                    },
                    hide_index=True
                )
                
                # Download option
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results CSV",
                    data=csv,
                    file_name="json_match_results.csv",
                    mime="text/csv"
                )
                
                # Visualization options based on result count
                if len(results_df) > 1:  # Multiple results - batch visualization
                    display_styled_header("Results Analysis")
                    
                    # Create visualizations
                    hist_fig, bar_fig, scatter_fig, pie_fig, summary_stats = create_batch_visualization(results_df)
                    
                    # Display summary statistics
                    st.subheader("Summary Statistics")
                    display_summary_metrics(summary_stats)
                    
                    # Display visualizations
                    st.subheader("Score Distribution")
                    st.plotly_chart(hist_fig, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(bar_fig, use_container_width=True)
                    with col2:
                        st.plotly_chart(pie_fig, use_container_width=True)
                    
                    st.plotly_chart(scatter_fig, use_container_width=True)
                
                elif len(results_df) == 1:  # Single result - individual visualization
                    # Display gauge for single result
                    score = results_df.iloc[0]['score']
                    st.subheader("Score Visualization")
                    gauge_fig = create_score_gauge(score)
                    st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    # Display explanation if available
                    if explanations and using_ml:
                        explanation = explanations[0]
                        importance_fig = visualize_feature_importance(explanation)
                        if importance_fig:
                            st.plotly_chart(importance_fig, use_container_width=True)
                
                # Display detailed results
                with st.expander("Detailed Results (JSON)"):
                    st.json(results)
                
                # Display explanations if available
                if explanations and using_ml:
                    with st.expander("Match Explanations"):
                        for explanation in explanations:
                            job_app_id = explanation.get('job_app_id')
                            st.write(f"**Job Application ID: {job_app_id}**")
                            
                            features = explanation.get('features', [])
                            
                            # Separate positive and negative factors
                            pos_features = [f for f in features if f['impact'] > 0]
                            neg_features = [f for f in features if f['impact'] < 0]
                            
                            # Top positive factors
                            if pos_features:
                                st.write("**Strengths:**")
                                for i, feature in enumerate(pos_features[:5]):
                                    impact = feature['impact'] * 100  # Scale to percentage points
                                    st.write(f"• {feature['feature']}: +{impact:.1f} points")
                            
                            # Top negative factors
                            if neg_features:
                                st.write("**Areas for Improvement:**")
                                for i, feature in enumerate(neg_features[:5]):
                                    impact = abs(feature['impact']) * 100  # Scale to percentage points
                                    st.write(f"• {feature['feature']}: -{impact:.1f} points")
                            
                            st.write("---")
            else:
                st.error("Failed to parse JSON input")
        
        except Exception as e:
            st.error(f"Error processing data: {e}")
            st.help("Make sure your JSON is properly formatted. Use valid quote characters and check for missing commas.")