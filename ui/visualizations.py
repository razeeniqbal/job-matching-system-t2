# ui/visualizations.py
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils.scoring import get_score_color

def create_radar_chart(values, categories, title="Skills Match"):
    """Create a radar chart for visualizing different aspects of the match."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Match Score',
        line_color='rgba(31, 119, 180, 0.8)',
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        title=title,
        height=350
    )
    
    return fig

def create_score_gauge(score, title="Match Score"):
    """Create a gauge chart to visualize the score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': get_score_color(score)},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 70], 'color': "lightblue"},
                {'range': [70, 100], 'color': "lavender"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

# ui/visualizations.py - Updated visualize_feature_importance function

def visualize_feature_importance(explanation, title="Feature Importance"):
    """Create a horizontal bar chart of feature importance with improved naming."""
    if not explanation or 'features' not in explanation:
        return None
        
    features = explanation['features']
    
    # Improved feature name mapping
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
    
    # Extract feature names and importance values
    feature_names = []
    for feature in features[:10]:
        raw_name = str(feature['feature'])
        # Use the mapped name if available, otherwise use the raw name
        feature_names.append(feature_name_mapping.get(raw_name, raw_name))
    
    importance = [float(feature['impact'] * 100) for feature in features[:10]]
    
    # Create a Plotly bar chart
    colors = ['#2196F3' if imp > 0 else '#F44336' for imp in importance]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=feature_names,
        x=importance,
        orientation='h',
        marker_color=colors
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Impact on Score (%)",
        yaxis_title="Feature",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_batch_visualization(results_df):
    """Create visualizations for batch processing results."""
    
    # Ensure all numeric values are properly typed
    results_df['score'] = results_df['score'].astype(float)
    
    # 1. Create a histogram of scores
    hist_fig = px.histogram(
        results_df, 
        x="score", 
        nbins=10, 
        title="Distribution of Match Scores",
        labels={"score": "Match Score (%)", "count": "Number of Applications"},
        color_discrete_sequence=['#1E88E5']
    )
    hist_fig.update_layout(bargap=0.1)
    
    # 2. Create a bar chart of top applications
    top_df = results_df.sort_values("score", ascending=False).head(10)
    bar_fig = px.bar(
        top_df,
        x="score",
        y="job_app_id",
        orientation='h',
        title="Top 10 Matches",
        labels={"score": "Match Score (%)", "job_app_id": "Application ID"},
        color="score",
        color_continuous_scale=px.colors.sequential.Blues
    )
    bar_fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    # 3. Create a scatter plot to show the distribution
    scatter_fig = px.scatter(
        results_df,
        x="job_app_id",
        y="score",
        title="Match Scores by Application ID",
        labels={"score": "Match Score (%)", "job_app_id": "Application ID"},
        color="score",
        color_continuous_scale=px.colors.sequential.Viridis,
        size="score",
        size_max=15
    )
    
    # Create a summary box
    summary_stats = {
        "Average Score": float(results_df["score"].mean()),
        "Highest Score": float(results_df["score"].max()),
        "Lowest Score": float(results_df["score"].min()),
        "Number of Applications": int(len(results_df)),
        "Applications >= 70%": int(len(results_df[results_df["score"] >= 70])),
        "Applications < 50%": int(len(results_df[results_df["score"] < 50]))
    }
    
    # Add ranges info
    score_ranges = [
        {"name": "Excellent (90-100%)", "min": 90, "max": 100, "color": "#4CAF50"},
        {"name": "Good (70-89%)", "min": 70, "max": 90, "color": "#2196F3"},
        {"name": "Partial (50-69%)", "min": 50, "max": 70, "color": "#FFC107"},
        {"name": "Poor (0-49%)", "min": 0, "max": 50, "color": "#F44336"}
    ]
    
    ranges_df = pd.DataFrame([
        {
            "range": r["name"],
            "count": int(len(results_df[(results_df["score"] >= r["min"]) & (results_df["score"] < r["max"])]))
        } 
        for r in score_ranges
    ])
    
    # Create a pie chart of score ranges
    pie_fig = px.pie(
        ranges_df, 
        values='count', 
        names='range', 
        title="Distribution by Match Quality",
        color='range',
        color_discrete_map={
            "Excellent (90-100%)": "#4CAF50",
            "Good (70-89%)": "#2196F3", 
            "Partial (50-69%)": "#FFC107",
            "Poor (0-49%)": "#F44336"
        }
    )
    
    return hist_fig, bar_fig, scatter_fig, pie_fig, summary_stats