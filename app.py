import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MHT CET Analytics Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #E0E6ED;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00E5FF;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8A99AD;
        text-transform: uppercase;
    }
    .main-title {
        background: linear-gradient(90deg, #00E5FF 0%, #00E676 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.3rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load('MHT_CET_model_under_19MB.pkl')
        return model
    except Exception as e:
        st.warning(f"Model load warning: {e}. Fallback logic enabled.")
        return None

model = load_model()

# -----------------------------------------------------------------------------
# SIDEBAR & INPUTS
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">MHT CET Predictive Intelligence Hub</div>', unsafe_allow_html=True)
st.markdown("##### Performance analytics and percentile estimations for Maharashtra CET")
st.markdown("---")

st.sidebar.header("🎯 Marks Inputs")
phy_score = st.sidebar.slider("Physics Marks (out of 50)", 0, 50, 38)
chem_score = st.sidebar.slider("Chemistry Marks (out of 50)", 0, 50, 41)
math_score = st.sidebar.slider("Mathematics Marks (out of 100)", 0, 100, 78)

total_score = phy_score + chem_score + math_score

# -----------------------------------------------------------------------------
# PREDICTION LOGIC
# -----------------------------------------------------------------------------
predicted_percentile = 0.0

if model is not None:
    try:
        # Pass inputs as standard DataFrame
        input_data = pd.DataFrame([[phy_score, chem_score, math_score]], 
                                  columns=['Physics', 'Chemistry', 'Mathematics'])
        prediction = model.predict(input_data)
        predicted_percentile = float(prediction[0])
    except Exception:
        # Alternative attempt in case model expects array
        try:
            prediction = model.predict([[phy_score, chem_score, math_score]])
            predicted_percentile = float(prediction[0])
        except Exception:
            predicted_percentile = min(99.99, (total_score / 200) ** 1.3 * 100)
else:
    predicted_percentile = min(99.99, (total_score / 200) ** 1.3 * 100)

# -----------------------------------------------------------------------------
# DASHBOARD
# -----------------------------------------------------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Score Metrics")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Calculated Score</div>
        <div class="metric-value">{total_score} / 200</div>
    </div>
    <br>
    <div class="metric-card">
        <div class="metric-label">Estimated Percentile</div>
        <div class="metric-value">{predicted_percentile:.2f} %ile</div>
    </div>
    """, unsafe_allow_html=True)

    fig_pie = px.pie(
        values=[phy_score, chem_score, math_score],
        names=['Physics', 'Chemistry', 'Mathematics'],
        hole=0.6,
        color_discrete_sequence=['#00E5FF', '#00E676', '#7C4DFF'],
        title="Subject Score Breakdown"
    )
    fig_pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E6ED'),
        margin=dict(t=40, b=0, l=0, r=0)
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("📈 Percentile Analytics")
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_percentile,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Competitive Gauge", 'font': {'color': '#E0E6ED', 'size': 16}},
        number={'suffix': " %ile", 'font': {'color': '#00E5FF', 'size': 30}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "#8A99AD"},
            'bar': {'color': "#00E5FF"},
            'bgcolor': "rgba(255,255,255,0.05)",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255, 82, 82, 0.2)'},
                {'range': [50, 85], 'color': 'rgba(255, 177, 66, 0.2)'},
                {'range': [85, 100], 'color': 'rgba(0, 230, 118, 0.2)'}
            ]
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E6ED'),
        height=250,
        margin=dict(t=40, b=10, l=30, r=30)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    x_marks = np.linspace(0, 200, 100)
    y_percentile = (x_marks / 200) ** 1.3 * 100
    
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=x_marks, y=y_percentile, mode='lines', name='Benchmark', line=dict(color='#00E5FF')))
    fig_curve.add_trace(go.Scatter(x=[total_score], y=[predicted_percentile], mode='markers', name='Your Standing', marker=dict(color='#00E676', size=12)))
    fig_curve.update_layout(
        title="Score vs Percentile Curve",
        xaxis_title="Raw Marks",
        yaxis_title="Percentile",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E6ED'),
        height=230,
        margin=dict(t=40, b=20, l=0, r=0)
    )
    st.plotly_chart(fig_curve, use_container_width=True)
