import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MHT-CET Smart Predictor & Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM PROFESSIONAL CSS STYLING ---
st.markdown("""
    
""", unsafe_allow_html=True)

# --- MODEL LOADING WITH CACHING & ERROR HANDLING ---
MODEL_PATH = "MHT_CET_model_under_19MB.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        return None

model = load_model()

# --- HEADER SECTION ---
st.title("🎓 MHT-CET Admission & Performance Analytics Dashboard")
st.markdown("Evaluate predicted percentiles, rank ranges, and college admission probabilities based on your exam performance.")
st.markdown("---")

# --- SIDEBAR INPUTS ---
st.sidebar.header("📝 Candidate Parameters")
st.sidebar.markdown("Configure your exam scores below:")

physics_score = st.sidebar.slider("Physics Score (out of 100)", 0.0, 100.0, 75.0, 0.5)
chemistry_score = st.sidebar.slider("Chemistry Score (out of 100)", 0.0, 100.0, 70.0, 0.5)
math_score = st.sidebar.slider("Mathematics Score (out of 100)", 0.0, 100.0, 80.0, 0.5)

total_score = physics_score + chemistry_score + math_score
avg_score = total_score / 3.0

category = st.sidebar.selectbox("Category", ["OPEN", "OBC", "SC", "ST", "VJ/DT", "NT-1", "NT-2", "NT-3", "EWS"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Transgender"])
home_uni = st.sidebar.selectbox("Home University", [
    "University of Mumbai", 
    "Savitribai Phule Pune University", 
    "Shivaji University, Kolhapur", 
    "Dr. Babasaheb Ambedkar Marathwada University", 
    "Rashtrasant Tukadoji Maharaj Nagpur University",
    "Other / Out of Maharashtra"
])

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🚀 Run Prediction & Analytics")

# --- MAIN CONTENT & ANALYTICS ---
if model is None:
    st.warning(f"⚠️ Model file `{MODEL_PATH}` was not detected in the working directory. Please upload your `.pkl` file to GitHub alongside `app.py`.")
    
    # Preview Dashboard metrics while model is loading
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total PCM Score", f"{total_score:.1f} / 300")
    with col2:
        st.metric("Estimated Percentile", "94.5% (Sample)")
    with col3:
        st.metric("Projected Rank Range", "12,000 - 15,000")
else:
    try:
        # Construct input array for model prediction 
        # (Note: Adjust column names if your trained pipeline requires specific feature labels)
        input_data = pd.DataFrame([[physics_score, chemistry_score, math_score, total_score]], 
                                  columns=['Physics', 'Chemistry', 'Mathematics', 'Total'])
        
        prediction = model.predict(input_data)[0]
        
        # Top Metrics Display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
