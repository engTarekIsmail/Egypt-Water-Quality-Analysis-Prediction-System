# Professional Streamlit Water Quality Prediction App

import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier
import os

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Egypt Water Quality AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# CUSTOM CSS
# =========================

st.markdown(
    """
    <style>

    .main {
        background-color: #0E1117;
    }

    .title {
        font-size: 42px;
        font-weight: bold;
        color: #00D4FF;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle {
        font-size: 18px;
        color: #CFCFCF;
        text-align: center;
        margin-bottom: 40px;
    }

    .metric-card {
        background-color: #161B22;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #2A2F3A;
        text-align: center;
    }

    .section-title {
        font-size: 24px;
        font-weight: bold;
        color: #00D4FF;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .stButton>button {
        width: 100%;
        background-color: #00D4FF;
        color: black;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        height: 55px;
        border: none;
    }

    .stButton>button:hover {
        background-color: #00B8E6;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# LOAD MODEL
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "water_quality_model.pkl")

model = joblib.load(model_path)
feature_names = joblib.load("feature_names.pkl")

# =========================
# HEADER
# =========================

st.markdown(
    '<div class="title">💧 Egypt Water Quality Prediction System</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">AI-powered system for predicting drinking water safety using machine learning</div>',
    unsafe_allow_html=True,
)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("📘 About Project")

st.sidebar.info("""
    This AI system predicts whether water is safe for drinking based on laboratory water quality measurements.

    Developed using:
    - Machine Learning
    - Random Forest Classifier
    - Streamlit
    - Environmental Data Analysis
    """)

st.sidebar.success("Model Accuracy: 99.3%")

# =========================
# INPUT SECTIONS
# =========================

st.markdown(
    '<div class="section-title">🧪 Water Test Parameters</div>', unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

# =========================
# COLUMN 1
# =========================

with col1:

    pH = st.number_input(
        "pH",
        min_value=0.0,
        max_value=14.0,
        value=7.9,
        step=0.1,
        help="WHO Safe Range: 6.5 - 8.5",
    )

    turbidity_NTU = st.number_input(
        "Turbidity (NTU)",
        min_value=0.0,
        value=0.48,
        step=0.01,
        help="WHO Safe Limit: <= 5 NTU",
    )

    TDS_mg_L = st.number_input(
        "TDS (mg/L)",
        min_value=0,
        value=330,
        step=10,
        help="WHO Safe Limit: <= 500 mg/L",
    )

    chlorine_mg_L = st.number_input(
        "Chlorine (mg/L)",
        min_value=0.0,
        value=0.67,
        step=0.01,
        help="Recommended: 0.2 - 0.8 mg/L",
    )

    EC_uS_cm = st.number_input(
        "Electrical Conductivity (uS/cm)",
        min_value=0,
        value=563,
        step=10,
        help="Lower values indicate cleaner water",
    )

# =========================
# COLUMN 2
# =========================

with col2:

    nitrate_mg_L = st.number_input(
        "Nitrate (mg/L)",
        min_value=0.0,
        value=2.35,
        step=0.01,
        help="WHO Safe Limit: <= 5 mg/L",
    )

    nitrite_mg_L = st.number_input(
        "Nitrite (mg/L)",
        min_value=0.0,
        value=0.017,
        step=0.001,
        help="WHO Safe Limit: <= 0.1 mg/L",
    )

    ammonia_mg_L = st.number_input(
        "Ammonia (mg/L)",
        min_value=0.0,
        value=0.087,
        step=0.001,
        help="Recommended: <= 0.5 mg/L",
    )

    iron_Fe_mg_L = st.number_input(
        "Iron Fe (mg/L)",
        min_value=0.0,
        value=0.097,
        step=0.001,
        help="WHO Safe Limit: <= 0.3 mg/L",
    )

    manganese_Mn_mg_L = st.number_input(
        "Manganese Mn (mg/L)",
        min_value=0.0,
        value=0.015,
        step=0.001,
        help="WHO Safe Limit: <= 0.1 mg/L",
    )

# =========================
# COLUMN 3
# =========================

with col3:

    sulfate_mg_L = st.number_input(
        "Sulfate (mg/L)",
        min_value=0,
        value=40,
        step=10,
        help="WHO Safe Limit: <= 250 mg/L",
    )

    hardness_mg_L = st.number_input(
        "Hardness (mg/L)",
        min_value=0,
        value=170,
        step=10,
        help="Recommended: <= 300 mg/L",
    )

    chloride_mg_L = st.number_input(
        "Chloride (mg/L)",
        min_value=0,
        value=43,
        step=10,
        help="WHO Safe Limit: <= 250 mg/L",
    )

    coliform_CFU_per_100mL = st.number_input(
        "Coliform (CFU/100mL)",
        min_value=0,
        value=0,
        step=1,
        help="Safe Drinking Water = 0",
    )

# =========================
# PREDICTION
# =========================

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Predict Water Quality"):

    input_dict = {
        "pH": pH,
        "turbidity_NTU": turbidity_NTU,
        "TDS_mg_L": TDS_mg_L,
        "chlorine_mg_L": chlorine_mg_L,
        "EC_uS_cm": EC_uS_cm,
        "nitrate_mg_L": nitrate_mg_L,
        "nitrite_mg_L": nitrite_mg_L,
        "ammonia_mg_L": ammonia_mg_L,
        "iron_Fe_mg_L": iron_Fe_mg_L,
        "manganese_Mn_mg_L": manganese_Mn_mg_L,
        "sulfate_mg_L": sulfate_mg_L,
        "hardness_mg_L": hardness_mg_L,
        "chloride_mg_L": chloride_mg_L,
        "coliform_CFU_per_100mL": coliform_CFU_per_100mL,
    }

    # Create DataFrame
    input_data = pd.DataFrame([input_dict])

    # Ensure correct column order
    input_data = input_data[feature_names]

    # Debug display
    st.write("Input Data:")
    st.dataframe(input_data)

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probabilities
    probability = model.predict_proba(input_data)[0]

    st.markdown(
        '<div class="section-title">📊 Prediction Result</div>', unsafe_allow_html=True
    )

    colA, colB = st.columns(2)

    # =========================
    # RESULT COLUMN
    # =========================

    with colA:

        if prediction == 1:

            st.success("✅ Water is SAFE for drinking")

            st.metric("Confidence", f"{probability[1]*100:.2f}%")

        else:

            st.error("❌ Water is NOT SAFE for drinking")

            st.metric("Confidence", f"{probability[0]*100:.2f}%")

    # =========================
    # POLLUTION REASONS
    # =========================

    with colB:

        st.markdown("### ⚠ Potential Pollution Indicators")

        reasons = []

        if turbidity_NTU > 5:
            reasons.append("High Turbidity")

        if nitrate_mg_L > 50:
            reasons.append("High Nitrate")

        if ammonia_mg_L > 0.5:
            reasons.append("High Ammonia")

        if iron_Fe_mg_L > 0.3:
            reasons.append("High Iron")

        if manganese_Mn_mg_L > 0.1:
            reasons.append("High Manganese")

        if coliform_CFU_per_100mL > 0:
            reasons.append("Bacterial Contamination")
    # =========================
    # SAFE WATER
    # =========================

    if prediction == 1:

        st.success("No major contamination indicators detected")

    # =========================
    # UNSAFE WATER
    # =========================

    else:

        if len(reasons) > 0:

            for reason in reasons:
                st.warning(reason)

        else:

            st.warning("Unsafe water detected by AI model")
# =========================
# FOOTER
# =========================

st.markdown("---")

st.markdown(
    """
    <center>
    Developed for Environmental Water Quality Monitoring using Machine Learning 🚀
    </center>
    """,
    unsafe_allow_html=True,
)
