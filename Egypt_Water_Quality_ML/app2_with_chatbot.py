# Professional Streamlit Water Quality Prediction App + GPT Chatbot

import streamlit as st
import pandas as pd
import joblib
from openai import OpenAI
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
# OPENAI CLIENT
# =========================



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

    .chat-container {
        background-color: #161B22;
        border-radius: 15px;
        border: 1px solid #2A2F3A;
        padding: 15px;
        max-height: 400px;
        overflow-y: auto;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# LOAD MODEL
# =========================

model = joblib.load("water_quality_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# =========================
# SESSION STATE INIT
# =========================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "water_context" not in st.session_state:
    st.session_state.water_context = None

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
# SIDEBAR - About + Chatbot
# =========================

st.sidebar.title("📘 About Project")

st.sidebar.info("""
    This AI system predicts whether water is safe for drinking based on laboratory water quality measurements.

    Developed using:
    - Machine Learning
    - XGBoost
    - Streamlit
    - Environmental Data Analysis
    """)

st.sidebar.success("Model Accuracy: 99.3%")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🤖 Water Quality Assistant")
st.sidebar.markdown("اسأل عن نتيجة التحليل أو أي سؤال عن جودة المياه")

# Display chat history in sidebar
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.sidebar.chat_message("user").write(msg["content"])
    else:
        st.sidebar.chat_message("assistant").write(msg["content"])

# Chat input
user_question = st.sidebar.chat_input("اكتب سؤالك هنا...")

if user_question:
    # Build system prompt with water context if available
    if st.session_state.water_context:
        ctx = st.session_state.water_context
        system_prompt = f"""أنت مساعد متخصص في جودة المياه وتعمل ضمن نظام تحليل مياه الشرب المصري.

المستخدم أجرى تحليلاً لعينة مياه وكانت النتائج كالتالي:

القيم المُدخلة:
- pH: {ctx['pH']}
- العكارة (Turbidity): {ctx['turbidity_NTU']} NTU
- المواد الصلبة الذائبة (TDS): {ctx['TDS_mg_L']} mg/L
- الكلور (Chlorine): {ctx['chlorine_mg_L']} mg/L
- التوصيل الكهربائي (EC): {ctx['EC_uS_cm']} uS/cm
- النترات (Nitrate): {ctx['nitrate_mg_L']} mg/L
- النتريت (Nitrite): {ctx['nitrite_mg_L']} mg/L
- الأمونيا (Ammonia): {ctx['ammonia_mg_L']} mg/L
- الحديد (Iron): {ctx['iron_Fe_mg_L']} mg/L
- المنجنيز (Manganese): {ctx['manganese_Mn_mg_L']} mg/L
- الكبريتات (Sulfate): {ctx['sulfate_mg_L']} mg/L
- الصلابة (Hardness): {ctx['hardness_mg_L']} mg/L
- الكلوريد (Chloride): {ctx['chloride_mg_L']} mg/L
- البكتيريا القولونية (Coliform): {ctx['coliform_CFU_per_100mL']} CFU/100mL

نتيجة نموذج الذكاء الاصطناعي: {'المياه صالحة للشرب ✅' if ctx['prediction'] == 1 else 'المياه غير صالحة للشرب ❌'}
نسبة الثقة: {ctx['confidence']:.2f}%
المؤشرات التحذيرية المكتشفة: {', '.join(ctx['reasons']) if ctx['reasons'] else 'لا توجد'}

أجب على أسئلة المستخدم بشكل احترافي ومفصل. يمكنك الإجابة بالعربية أو الإنجليزية حسب لغة السؤال.
إذا كانت المياه غير صالحة، اشرح الأسباب والمخاطر الصحية وكيفية المعالجة.
إذا كانت صالحة، يمكنك تقديم نصائح للحفاظ على جودة المياه."""
    else:
        system_prompt = """أنت مساعد متخصص في جودة المياه وتعمل ضمن نظام تحليل مياه الشرب المصري.
أجب على أسئلة المستخدم المتعلقة بجودة المياه، معايير منظمة الصحة العالمية، والمؤشرات الكيميائية والبيولوجية.
يمكنك الإجابة بالعربية أو الإنجليزية حسب لغة السؤال.
ملاحظة: المستخدم لم يُجرِ تحليلاً بعد. شجّعه على إدخال قيم العينة أولاً للحصول على تحليل دقيق."""

    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    # Call GPT
    try:
        messages_to_send = [{"role": "system", "content": system_prompt}]
        # Add last 6 messages for context (avoid token overflow)
        for msg in st.session_state.chat_history[-6:]:
            messages_to_send.append({"role": msg["role"], "content": msg["content"]})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_to_send,
            max_tokens=800,
            temperature=0.7,
        )

        assistant_reply = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})

    except Exception as e:
        error_msg = f"❌ خطأ في الاتصال بـ ChatGPT: {str(e)}"
        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

    st.rerun()

# Clear chat button
if st.sidebar.button("🗑️ مسح المحادثة"):
    st.session_state.chat_history = []
    st.rerun()

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

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probabilities
    probability = model.predict_proba(input_data)[0]

    # Pollution reasons
    reasons = []
    if turbidity_NTU > 5:
        reasons.append("High Turbidity (عكارة عالية)")
    if nitrate_mg_L > 50:
        reasons.append("High Nitrate (نترات عالية)")
    if ammonia_mg_L > 0.5:
        reasons.append("High Ammonia (أمونيا عالية)")
    if iron_Fe_mg_L > 0.3:
        reasons.append("High Iron (حديد عالي)")
    if manganese_Mn_mg_L > 0.1:
        reasons.append("High Manganese (منجنيز عالي)")
    if coliform_CFU_per_100mL > 0:
        reasons.append("Bacterial Contamination (تلوث بكتيري)")

    # Save context for chatbot
    confidence = probability[1] * 100 if prediction == 1 else probability[0] * 100
    st.session_state.water_context = {
        **input_dict,
        "prediction": int(prediction),
        "confidence": confidence,
        "reasons": reasons,
    }

    # Auto-trigger GPT analysis
    if prediction == 1:
        auto_prompt = "حلل نتيجة عينة المياه دي وقولي إيه تقييمك للقيم الموجودة."
    else:
        auto_prompt = "المياه غير صالحة، اشرح لي بالتفصيل ليه وإيه المخاطر الصحية وطرق المعالجة."

    ctx = st.session_state.water_context
    system_prompt = f"""أنت مساعد متخصص في جودة المياه وتعمل ضمن نظام تحليل مياه الشرب المصري.

القيم المُدخلة:
- pH: {ctx['pH']}
- العكارة (Turbidity): {ctx['turbidity_NTU']} NTU
- المواد الصلبة الذائبة (TDS): {ctx['TDS_mg_L']} mg/L
- الكلور (Chlorine): {ctx['chlorine_mg_L']} mg/L
- التوصيل الكهربائي (EC): {ctx['EC_uS_cm']} uS/cm
- النترات (Nitrate): {ctx['nitrate_mg_L']} mg/L
- النتريت (Nitrite): {ctx['nitrite_mg_L']} mg/L
- الأمونيا (Ammonia): {ctx['ammonia_mg_L']} mg/L
- الحديد (Iron): {ctx['iron_Fe_mg_L']} mg/L
- المنجنيز (Manganese): {ctx['manganese_Mn_mg_L']} mg/L
- الكبريتات (Sulfate): {ctx['sulfate_mg_L']} mg/L
- الصلابة (Hardness): {ctx['hardness_mg_L']} mg/L
- الكلوريد (Chloride): {ctx['chloride_mg_L']} mg/L
- البكتيريا القولونية (Coliform): {ctx['coliform_CFU_per_100mL']} CFU/100mL

نتيجة النموذج: {'المياه صالحة للشرب ✅' if ctx['prediction'] == 1 else 'المياه غير صالحة للشرب ❌'}
نسبة الثقة: {ctx['confidence']:.2f}%
المؤشرات التحذيرية: {', '.join(ctx['reasons']) if ctx['reasons'] else 'لا توجد'}

أجب بشكل احترافي ومفصل باللغة العربية."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": auto_prompt},
            ],
            max_tokens=800,
            temperature=0.7,
        )
        auto_reply = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "user", "content": "📊 تحليل تلقائي بعد الـ Prediction"})
        st.session_state.chat_history.append({"role": "assistant", "content": auto_reply})
    except Exception as e:
        pass  # Silent fail for auto analysis

    # =========================
    # DISPLAY RESULTS
    # =========================

    st.markdown(
        '<div class="section-title">📊 Prediction Result</div>', unsafe_allow_html=True
    )

    colA, colB = st.columns(2)

    with colA:

        if prediction == 1:
            st.success("✅ Water is SAFE for drinking")
            st.metric("Confidence", f"{probability[1]*100:.2f}%")
        else:
            st.error("❌ Water is NOT SAFE for drinking")
            st.metric("Confidence", f"{probability[0]*100:.2f}%")

    with colB:

        st.markdown("### ⚠ Potential Pollution Indicators")

        if prediction == 1:
            st.success("No major contamination indicators detected")
        else:
            if len(reasons) > 0:
                for reason in reasons:
                    st.warning(reason)
            else:
                st.warning("Unsafe water detected by AI model")

    # Prompt user to check chatbot
    st.info("""
🤖 **AI Water Quality Assistant**

The water sample has been analyzed automatically.

Check the Sidebar — You can now ask questions such as:

- Why was this sample classified as unsafe?
- What are the main pollutants?
- What are the health risks?
- How can the water be treated?
- Does it comply with WHO standards?
""")

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
