# ui/single_match_tab.py
import streamlit as st
import pandas as pd
from models.xgboost_job_matcher import XGBoostJobMatcher
from models.feature_engineering import get_raw_match_scores
from config.settings import EDUCATION_OPTIONS
from utils.scoring import calculate_component_scores, get_score_color
from ui.visualizations import create_radar_chart, create_score_gauge, visualize_feature_importance

def load_model():
    """Load the trained XGBoost model if available, otherwise return None."""
    try:
        model = XGBoostJobMatcher.load()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def calculate_match_score(job_app):
    """Calculate match score using either ML model or rule-based approach."""
    model = load_model()
    
    # If model exists, use ML prediction
    if model:
        predictions = model.predict([job_app])
        score = predictions[0]['score'] if predictions else 0
        explanation = model.explain([job_app])[0] if predictions else None
        return score, explanation, True  # True indicates ML model was used
    
    # Fallback to rule-based scoring
    else:
        results = get_raw_match_scores([job_app])
        score = results[0]['score'] if results else 0
        return score, None, False  # False indicates rule-based approach was used

def render_single_match_tab():
    """Render the single match tab UI."""
    st.markdown("<h2 class='sub-header'>Calculate Match Score</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # Job Requirements
    with col1:
        st.subheader("Job Requirements")
        
        # Core Skills
        st.write("**Core Skills Required** (50% weight)")
        job_skills = st.text_area(
            "Enter required skills (comma-separated)",
            value="Python, Machine Learning, SQL",
            key="job_skills"
        )
        job_skills_list = [skill.strip() for skill in job_skills.split(",") if skill.strip()]
        
        # Experience
        st.write("**Minimum Experience** (20% weight)")
        min_experience = st.number_input(
            "Years of experience required",
            min_value=0,
            max_value=20,
            value=3,
            step=1,
            key="min_exp"
        )
        
        # Education
        st.write("**Education Level** (15% weight)")
        education_required = st.selectbox(
            "Minimum education required",
            options=EDUCATION_OPTIONS,
            index=3,  # Default to Bachelor's
            key="req_edu"
        )
        
        # Certifications
        st.write("**Required Certifications** (15% weight)")
        job_certs = st.text_area(
            "Enter required certifications (comma-separated)",
            value="AWS Certified",
            key="job_certs"
        )
        job_certs_list = [cert.strip() for cert in job_certs.split(",") if cert.strip()]
    
    # Talent Profile
    with col2:
        st.subheader("Talent Profile")
        
        # Skills
        st.write("**Candidate Skills**")
        talent_skills = st.text_area(
            "Enter candidate skills (comma-separated)",
            value="Python, Machine Learning, Deep Learning, SQL",
            key="talent_skills"
        )
        talent_skills_list = [skill.strip() for skill in talent_skills.split(",") if skill.strip()]
        
        # Experience
        st.write("**Candidate Experience**")
        experience = st.number_input(
            "Years of experience",
            min_value=0,
            max_value=30,
            value=4,
            step=1,
            key="talent_exp"
        )
        
        # Education
        st.write("**Candidate Education**")
        education = st.selectbox(
            "Highest education level",
            options=EDUCATION_OPTIONS,
            index=4,  # Default to Master's
            key="talent_edu"
        )
        
        # Certifications
        st.write("**Candidate Certifications**")
        talent_certs = st.text_area(
            "Enter certifications (comma-separated)",
            value="AWS Certified, Google Cloud Certified",
            key="talent_certs"
        )
        talent_certs_list = [cert.strip() for cert in talent_certs.split(",") if cert.strip()]
    
    # Calculate button
    if st.button("Calculate Match Score", type="primary"):
        # Create job application object
        job_app = {
            "job_app_id": 999,  # Placeholder ID
            "job_requirements": {
                "core_skills": job_skills_list,
                "min_experience": min_experience,
                "education": education_required,
                "certifications": job_certs_list
            },
            "talent_profile": {
                "skills": talent_skills_list,
                "experience": experience,
                "education": education,
                "certifications": talent_certs_list
            }
        }
        
        # Calculate score
        score, explanation, is_ml = calculate_match_score(job_app)
        
        # Display results
        st.markdown("<h2 class='sub-header'>Match Results</h2>", unsafe_allow_html=True)
        
        # Improved visualization layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            
            # Show gauge chart
            gauge_fig = create_score_gauge(score)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Rating
            if score >= 90:
                match_quality = "Excellent Match"
                match_color = "#4CAF50"
            elif score >= 70:
                match_quality = "Good Match"
                match_color = "#2196F3"
            elif score >= 50:
                match_quality = "Partial Match"
                match_color = "#FFC107"
            else:
                match_quality = "Poor Match"
                match_color = "#F44336"
            
            st.markdown(f"<h3 style='text-align:center; color:{match_color};'>{match_quality}</h3>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # Calculate component scores
            component_scores = calculate_component_scores(
                job_app["job_requirements"],
                job_app["talent_profile"]
            )
            
            # Create radar chart
            radar_values = [
                component_scores['skill_match'] * 100, 
                component_scores['exp_match'] * 100, 
                component_scores['edu_match'] * 100, 
                component_scores['cert_match'] * 100
            ]
            radar_categories = ['Skills', 'Experience', 'Education', 'Certifications']
            radar_fig = create_radar_chart(radar_values, radar_categories, "Match Components")
            
            st.plotly_chart(radar_fig, use_container_width=True)
        
        # Show explanation if available (ML model)
        if explanation and is_ml:
            st.subheader("Key Factors Affecting Score")
            
            # Create feature importance chart
            importance_fig = visualize_feature_importance(explanation)
            if importance_fig:
                st.plotly_chart(importance_fig, use_container_width=True)
        
        # Basic breakdown (for both ML and rule-based)
        st.subheader("Score Breakdown")
        
        # Create a DataFrame for the breakdown
        breakdown_data = pd.DataFrame({
            'Component': ['Skills', 'Experience', 'Education', 'Certifications'],
            'Weight': [50, 20, 15, 15],
            'Match %': [
                component_scores['skill_match'] * 100, 
                component_scores['exp_match'] * 100, 
                component_scores['edu_match'] * 100, 
                component_scores['cert_match'] * 100
            ],
            'Weighted Score': [
                component_scores['skill_match'] * 50,
                component_scores['exp_match'] * 20,
                component_scores['edu_match'] * 15,
                component_scores['cert_match'] * 15
            ]
        })
        
        # Convert all columns to appropriate types to avoid PyArrow errors
        breakdown_data['Component'] = breakdown_data['Component'].astype(str)
        breakdown_data['Weight'] = breakdown_data['Weight'].astype(float)
        breakdown_data['Match %'] = breakdown_data['Match %'].astype(float)
        breakdown_data['Weighted Score'] = breakdown_data['Weighted Score'].astype(float)
        
        st.dataframe(
            breakdown_data,
            column_config={
                'Component': st.column_config.TextColumn('Component'),
                'Weight': st.column_config.NumberColumn('Weight %'),
                'Match %': st.column_config.ProgressColumn('Match %', format='%.1f%%', min_value=0, max_value=100),
                'Weighted Score': st.column_config.NumberColumn('Weighted Score', format='%.1f')
            },
            hide_index=True
        )