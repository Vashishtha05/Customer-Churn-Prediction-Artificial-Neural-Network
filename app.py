import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# ---------------------------------------------
# Page Config
# ---------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------
# Load Model + Encoders
# ---------------------------------------------
model = tf.keras.models.load_model("model.h5")

with open("label_encoder_gender.pkl","rb") as f:
    label_encoder_gender = pickle.load(f)

with open("onehot_encoder_geo.pkl","rb") as f:
    onehot_encoder_geo = pickle.load(f)

with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)

# ---------------------------------------------
# PREMIUM UI STYLING
# ---------------------------------------------
st.markdown("""
<style>

/* -------- Animated Gradient Background -------- */
.stApp {
    background:
        radial-gradient(circle at 10% 20%, rgba(59,130,246,0.15), transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(168,85,247,0.15), transparent 40%),
        linear-gradient(120deg,#f1f5f9,#eef2ff);
}

/* -------- Header -------- */
.header {
    font-size:42px;
    font-weight:700;
    text-align:center;
    color:#0f172a;
    margin-top:10px;
}

.subheader {
    text-align:center;
    color:#475569;
    margin-bottom:30px;
    font-size:18px;
}

/* -------- Glass Card -------- */
.card {
    background:rgba(255,255,255,0.65);
    backdrop-filter: blur(14px);
    padding:30px;
    border-radius:18px;
    box-shadow:0 10px 25px rgba(0,0,0,0.08);
    transition:0.3s;
}

/* Hover Glow */
.card:hover {
    box-shadow:0 14px 40px rgba(59,130,246,0.15);
}

/* -------- Button -------- */
div.stButton > button {
    background:linear-gradient(90deg,#3b82f6,#6366f1);
    color:white;
    border-radius:10px;
    height:3em;
    font-size:18px;
    border:none;
    transition:0.3s;
}

div.stButton > button:hover {
    transform:scale(1.03);
}

/* -------- Result Cards -------- */
.result-good {
    background:linear-gradient(90deg,#dcfce7,#bbf7d0);
    padding:22px;
    border-radius:14px;
    font-size:24px;
    text-align:center;
    color:#166534;
    font-weight:700;
    box-shadow:0 6px 18px rgba(0,0,0,0.06);
}

.result-bad {
    background:linear-gradient(90deg,#fee2e2,#fecaca);
    padding:22px;
    border-radius:14px;
    font-size:24px;
    text-align:center;
    color:#991b1b;
    font-weight:700;
    box-shadow:0 6px 18px rgba(0,0,0,0.06);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# HEADER
# ---------------------------------------------
st.markdown('<div class="header">Customer Churn AI Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Neural Network Powered Customer Risk Prediction</div>', unsafe_allow_html=True)

# ---------------------------------------------
# INPUT CARD
# ---------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    geography = st.selectbox("🌍 Geography", onehot_encoder_geo.categories_[0])
    gender = st.selectbox("👤 Gender", label_encoder_gender.classes_)
    age = st.slider("🎂 Age", 18, 92)

with col2:
    tenure = st.slider("📅 Tenure", 0, 10)
    num_products = st.slider("📦 Number of Products", 1, 4)
    has_cr_card = st.selectbox("💳 Has Credit Card", [0,1])

with col3:
    is_active_member = st.selectbox("⚡ Active Member", [0,1])
    balance = st.number_input("💰 Balance")
    credit_score = st.number_input("📊 Credit Score")
    estimated_salary = st.number_input("💵 Estimated Salary")

predict = st.button("🚀 Predict Churn Risk", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------
# PREDICTION
# ---------------------------------------------
if predict:

    input_data = pd.DataFrame({
        "CreditScore":[credit_score],
        "Gender":[label_encoder_gender.transform([gender])[0]],
        "Age":[age],
        "Tenure":[tenure],
        "Balance":[balance],
        "NumOfProducts":[num_products],
        "HasCrCard":[has_cr_card],
        "IsActiveMember":[is_active_member],
        "EstimatedSalary":[estimated_salary]
    })

    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
    )

    input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    prob = float(prediction[0][0])

    st.write("")
    st.progress(prob)

    if prob > 0.5:
        st.markdown(
            f'<div class="result-bad">🚨 High Churn Risk — Probability: {prob:.2f}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-good">✅ Low Churn Risk — Probability: {prob:.2f}</div>',
            unsafe_allow_html=True
        )
