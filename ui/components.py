# ui/components.py
import streamlit as st
import pandas as pd
import os

def display_styled_header(text, level=2):
    """Display a styled header with specified level."""
    if level == 1:
        st.markdown(f"<h1 class='main-header'>{text}</h1>", unsafe_allow_html=True)
    elif level == 2:
        st.markdown(f"<h2 class='sub-header'>{text}</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h{level} style='color:#0D47A1;'>{text}</h{level}>", unsafe_allow_html=True)

def display_metric_card(content, width=1):
    """Display content inside a styled metric card."""
    col = st.columns(width)[0]
    with col:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        content()  # Execute the content function inside the card
        st.markdown("</div>", unsafe_allow_html=True)

def display_match_quality_label(score):
    """Display a colored label indicating match quality."""
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
    
    st.markdown(f"<h3 style='text-align:center; color:{match_color};'>{match_quality}</h3>", 
                unsafe_allow_html=True)

def display_model_status():
    """Display model status in the sidebar."""
    model_available = os.path.exists("models/xgboost_model.joblib")
    
    if not model_available:
        st.sidebar.warning(
            "⚠️ XGBoost model not found. Using rule-based scoring instead. "
            "Train a model using `python scripts/train_model.py` to enable ML-based matching."
        )
    else:
        st.sidebar.success("✅ XGBoost model loaded successfully")
    
    return model_available

def display_summary_metrics(summary_stats):
    """Display summary statistics in metric components."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Score", f"{summary_stats['Average Score']:.1f}%")
    with col2:
        st.metric("Highest Score", f"{summary_stats['Highest Score']:.1f}%")
    with col3:
        st.metric("Lowest Score", f"{summary_stats['Lowest Score']:.1f}%")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Applications", summary_stats['Number of Applications'])
    with col2:
        st.metric("Good Matches (≥70%)", summary_stats['Applications >= 70%'])
    with col3:
        st.metric("Poor Matches (<50%)", summary_stats['Applications < 50%'])