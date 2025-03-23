import streamlit as st
import os
import nltk

# Set page configuration (MUST be the first Streamlit command)
st.set_page_config(
    page_title="Job Matching System",
    page_icon="🔍",
    layout="wide"
)

# Set up NLTK data path and download required data
def setup_nltk():
    # Create a data directory if it doesn't exist
    os.makedirs('nltk_data', exist_ok=True)
    
    # Set the NLTK data path to use our custom directory
    nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))
    
    # Download required NLTK data without using Streamlit commands
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print('Downloading NLTK punkt data...')
        nltk.download('punkt', download_dir='nltk_data')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print('Downloading NLTK wordnet data...')
        nltk.download('wordnet', download_dir='nltk_data')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print('Downloading NLTK stopwords data...')
        nltk.download('stopwords', download_dir='nltk_data')

# Run NLTK setup at startup
setup_nltk()

# Set Streamlit theme and styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .metric-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Import tab modules
from ui.single_match_tab import render_single_match_tab
from ui.batch_processing_tab import render_batch_processing_tab
from ui.json_input_tab import render_json_input_tab
from ui.model_info_tab import render_model_info_tab

def main():
    st.markdown("<h1 class='main-header'>🔍 Job Matching System</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>XGBoost-powered talent evaluation</h3>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This application uses XGBoost to match job candidates with job requirements. "
        "It evaluates skills, experience, education, and certifications to calculate a match score."
    )
    
    # Check if model is available
    model_available = os.path.exists("models/xgboost_model.joblib")
    
    if not model_available:
        st.sidebar.warning(
            "⚠️ XGBoost model not found. Using rule-based scoring instead. "
            "Train a model using `python scripts/train_model.py` to enable ML-based matching."
        )
    else:
        st.sidebar.success("✅ XGBoost model loaded successfully")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Batch Processing", "JSON Input","Single Match", "Model Information"])
    
    # Render each tab
    with tab1:
        render_batch_processing_tab()
    
    with tab2:
        render_json_input_tab()
        
    with tab3:
        render_single_match_tab()
    
    with tab4:
        render_model_info_tab()

if __name__ == "__main__":
    main()
