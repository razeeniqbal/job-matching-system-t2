# ui/batch_processing_tab.py
import streamlit as st
import pandas as pd
import json
from models.xgboost_job_matcher import XGBoostJobMatcher
from models.feature_engineering import get_raw_match_scores
from utils.data_loader import load_data_from_file
from ui.visualizations import create_batch_visualization
from ui.components import display_styled_header, display_summary_metrics

def load_model():
    """Load the trained XGBoost model if available, otherwise return None."""
    try:
        model = XGBoostJobMatcher.load()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def load_sample_data():
    """Load sample job application data from file."""
    return load_data_from_file("data/sample_jobs.json")

def render_batch_processing_tab():
    """Render the batch processing tab UI."""
    display_styled_header("Batch Process Job Applications")
    
    # Source selection
    data_source = st.radio(
        "Select data source:",
        ["Upload JSON file", "Use sample data"]
    )
    
    job_applications = []
    
    if data_source == "Upload JSON file":
        # File upload option
        st.write("Upload a JSON file with job applications to process in batch.")
        uploaded_file = st.file_uploader("Choose a JSON file", type="json")
        
        if uploaded_file is not None:
            try:
                # Load the data
                job_applications = json.load(uploaded_file)
                st.success(f"Successfully loaded {len(job_applications)} job applications")
            except Exception as e:
                st.error(f"Error parsing JSON file: {e}")
    else:
        # Use sample data
        job_applications = load_sample_data()
        if job_applications:
            st.success(f"Loaded {len(job_applications)} sample job applications")
        else:
            st.error("Failed to load sample data")
    
    # Display sample
    if job_applications:
        with st.expander("Preview Data (First Record)"):
            st.json(job_applications[0] if job_applications else {})
        
        # Process button
        if st.button("Process Job Applications", key="process_batch", type="primary"):
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
                file_name="job_match_results.csv",
                mime="text/csv"
            )
            
            # Enhanced Visualization for Batch Results
            if len(results_df) > 1:  # Only create visualizations if we have multiple results
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