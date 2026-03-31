import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Load model and tools
model = joblib.load(BASE_DIR / "model" / "demo_logistic_model.pkl")
scaler = joblib.load(BASE_DIR / "model" / "demo_scaler.pkl")
label_encoders = joblib.load(BASE_DIR / "model" / "demo_label_encoders.pkl")
feature_names = joblib.load(BASE_DIR / "model" / "demo_feature_names.pkl")

st.title("🏥 Patient Readmission Risk Predictor")

st.write("Enter patient details to estimate 30-day readmission risk.")

# --- INPUTS ---
age = st.selectbox("Age Group", ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                                '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'])

gender = st.selectbox("Gender", ["Male", "Female"])

time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 5)
num_lab_procedures = st.slider("Number of Lab Procedures", 0, 100, 40)
num_medications = st.slider("Number of Medications", 0, 50, 10)
num_procedures = st.slider("Number of Procedures", 0, 10, 1)
number_diagnoses = st.slider("Number of Diagnoses", 1, 10, 5)

# --- CREATE INPUT DATAFRAME ---
input_dict = {
    "age": age,
    "gender": gender,
    "time_in_hospital": time_in_hospital,
    "num_lab_procedures": num_lab_procedures,
    "num_medications": num_medications,
    "num_procedures": num_procedures,
    "number_diagnoses": number_diagnoses
}

input_df = pd.DataFrame([input_dict])

# --- ENCODE ---
for col in input_df.columns:
    if col in label_encoders:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col].astype(str))

# Fill missing columns (important!)
model_features = feature_names

input_df = input_df[model_features]

# --- SCALE ---
input_scaled = scaler.transform(input_df)

# --- PREDICT ---
if st.button("Predict Readmission Risk"):
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if probability < 0.35:
        st.success(f"✅ Low Risk of Readmission (Probability: {probability:.2f})")
        risk_level = "low"
    elif probability < 0.65:
        st.warning(f"⚠️ Moderate / Borderline Risk of Readmission (Probability: {probability:.2f})")
        risk_level = "moderate"
    else:
        st.error(f"🚨 High Risk of Readmission (Probability: {probability:.2f})")
        risk_level = "high"

    st.subheader("🧠 Risk Explanation")

    reasons = []

    if time_in_hospital > 7:
        reasons.append("Long hospital stay")
    if num_medications > 15:
        reasons.append("High number of medications")
    if number_diagnoses > 5:
        reasons.append("Multiple diagnoses")
    if num_lab_procedures > 50:
        reasons.append("High number of lab procedures")
    if age in ["[70-80)", "[80-90)", "[90-100)"]:
        reasons.append("Older age group")

    if reasons:
        st.write("Possible contributing factors:")
        for r in reasons:
            st.write(f"- {r}")
    else:
        st.write("No strong risk factors detected from selected inputs.")

    st.caption(
        "This is a simplified portfolio demo based on limited input features and should not be used for clinical decision-making."
    )