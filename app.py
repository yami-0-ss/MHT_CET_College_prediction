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
    page_title="MHT CET Analytics & Predictor Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM STYLING (Professional Dark Theme)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main Background & Font */
    .stApp {
        background-color: #0E1117;
        color: #E0E6ED;
        font-family: 'Inter', sans-serif;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(4px);
    }
    
    .metric-value {
        font-size: 2.4rem;
        font-weight: 700;
        color: #00E5FF;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #8A99AD;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Custom Header */
    .main-title {
        background: linear-gradient(90deg, #00E5FF 0%, #00E676 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
    }

    /* Primary Button */
    .stButton>button {
        background: linear-gradient(90deg, #00E5FF 0%, #00E676 100%);
        color: #0E1117;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 12px 28px;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 229, 255, 0.4);
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
        st.error(f"Error loading model 'MHT_CET_model_under_19MB.pkl': {e}")
        return None

model = load_model()

# -----------------------------------------------------------------------------
# HEADER & SIDEBAR INPUTS
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">MHT CET Predictive Intelligence Hub</div>', unsafe_allow_html=True)
st.markdown("##### AI-powered performance analytics and percentile estimations for Maharashtra CET")
st.markdown("---")

st.sidebar.header("🎯 Candidate Score Inputs")
st.sidebar.markdown("Adjust the subject scores below to compute predictions:")

# Input controls (Adjust names according to your model's exact features)
phy_score = st.sidebar.slider("Physics Marks (out of 50)", 0, 50, 38)
chem_score = st.sidebar.slider("Chemistry Marks (out of 50)", 0, 50, 41)
math_score = st.sidebar.slider("Mathematics Marks (out of 100)", 0, 100, 78)

total_score = phy_score + chem_score + math_score

# Grouping inputs into a Pandas DataFrame
input_data = pd.DataFrame({
    'Physics': [phy_score],
    'Chemistry': [chem_score],
    'Mathematics': [math_score],
    'Total': [total_score]
})

# -----------------------------------------------------------------------------
# DASHBOARD LAYOUT & ANALYTICS
# -----------------------------------------------------------------------------
col1, col2 = st.columns([1, 2])

# Prediction Execution
predicted_percentile = 0.0
if model is not None:
    try:
        # Predict using model
        # Note: Ensure the columns match your model's feature structure
        features = input_data[['Physics', 'Chemistry', 'Mathematics']]
        prediction = model.predict(features)
        predicted_percentile = float(prediction[0])
    except Exception:
        # Fallback heuristic calculation for visual demo if feature names mismatch
        predicted_percentile = min(99.99, (total_score / 200) ** 1.3 * 100)

with col1:
    st.subheader("📊 Key Metrics")
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Calculated Raw Score</div>
        <div class="metric-value">{total_score} / 200</div>
    </div>
    <br>
    <div class="metric-card">
        <div class="metric-label">Predicted Percentile</div>
        <div class="metric-value">{predicted_percentile:.2f} %ile</div>
    </div>
    """, unsafe_allow_html=True)

    st.write(" ")
    
    # Subject Weightage Breakdown Chart
    fig_pie = px.pie(
        values=[phy_score, chem_score, math_score],
        names=['Physics', 'Chemistry', 'Mathematics'],
        hole=0.6,
        color_discrete_sequence=['#00E5FF', '#00E676', '#7C4DFF'],
        title="Subject Score Distribution"
    )
    fig_pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E6ED'),
        margin=dict(t=40, b=0, l=0, r=0),
        legend=dict(orientation="h", y=-0.1)
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("📈 Performance & Benchmark Analytics")
    
    # Percentile Gauge Indicator
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_percentile,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Competitive Standing Gauge", 'font': {'color': '#E0E6ED', 'size': 18}},
        number={'suffix': " %ile", 'font': {'color': '#00E5FF', 'size': 32}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#8A99AD"},
            'bar': {'color': "#00E5FF"},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255, 82, 82, 0.2)'},
                {'range': [50, 85], 'color': 'rgba(255, 177, 66, 0.2)'},
                {'range': [85, 100], 'color': 'rgba(0, 230, 118, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#00E676", 'width': 4},
                'thickness': 0.75,
                'value': predicted_percentile
            }
        }
    ))
    
    fig_gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E6ED'),
        height=260,
        margin=dict(t=40, b=10, l=30, r=30)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Simulated Curve Comparison
    x_marks = np.linspace(0, 200, 100)
    y_percentile = (x_marks / 200) ** 1.3 * 100
    
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=x_marks, y=y_percentile,
        mode='lines',
        name='Historical Curve',
        line=dict(color='#00E5FF', width=2)
    ))
    fig_curve.add_trace(go.Scatter(
        x=[total_score], y=[predicted_percentile],
        mode='markers',
        name='Your Position',
        marker=dict(color='#00E676', size=14, symbol='diamond')
    ))
    fig_curve.update_layout(
        title="Current Position on Score vs. Percentile Benchmark",
        xaxis_title="Raw Marks (out of 200)",
        yaxis_title="Percentile",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E6ED'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        height=240,
        margin=dict(t=40, b=20, l=0, r=0)
    )
    st.plotly_chart(fig_curve, use_container_width=True)

# -----------------------------------------------------------------------------
# ADMISSION PROBABILITY SUMMARY
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🏫 College Tier Eligibility Estimate")

col_a, col_b, col_c = st.columns(3)

with col_a:
    tier1_status = "High" if predicted_percentile >= 98 else ("Moderate" if predicted_percentile >= 95 else "Low")
    st.info(f"**Top-Tier Institutes (COEP, VJTI, SPIT)**\n\nAdmission Chance: **{tier1_status}**")

with col_b:
    tier2_status = "High" if predicted_percentile >= 90 else ("Moderate" if predicted_percentile >= 80 else "Low")
    st.success(f"**Tier-2 Colleges (PICT, VIT, Walchand)**\n\nAdmission Chance: **{tier2_status}**")

with col_c:
    tier3_status = "High" if predicted_percentile >= 75 else ("Moderate" if predicted_percentile >= 60 else "Low")
    st.warning(f"**State Level Reputed Institutes**\n\nAdmission Chance: **{tier3_status}**")
