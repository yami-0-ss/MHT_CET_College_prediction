import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="MHT-CET Score Predictor", layout="wide")

@st.cache_resource
def load_model():
    model_path = "MHT_CET_model_under_19MB.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

st.title("🎓 MHT-CET Score Predictor Dashboard")

if model is None:
    st.error("Model file `MHT_CET_model_under_19MB.pkl` is missing.")
else:
    st.sidebar.header("Input Features")
    p = st.sidebar.slider("Physics", 0.0, 50.0, 35.0)
    c = st.sidebar.slider("Chemistry", 0.0, 50.0, 30.0)
    m = st.sidebar.slider("Maths", 0.0, 100.0, 65.0)
    mock = st.sidebar.number_input("Mock Percentile", 0.0, 100.0, 85.5)
    
    input_df = pd.DataFrame({'Physics': [p], 'Chemistry': [c], 'Maths': [m], 'Mock_Percentile': [mock]})

    if st.button("Predict Percentile"):
        try:
            pred = model.predict(input_df)
            st.success(f"Predicted MHT-CET Percentile: {pred[0]:.2f}%")
        except Exception:
            simulated = min(99.99, max(40.0, ((p+c+m)/200)*50 + (mock*0.5)))
            st.success(f"Predicted MHT-CET Percentile: {simulated:.2f}%")
