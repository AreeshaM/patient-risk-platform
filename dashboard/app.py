import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Patient Risk Platform", page_icon="🏥", layout="wide")

# Load model and data
model = pickle.load(open("readmission_model.pkl", "rb"))

st.title("🏥 AI-Powered Patient Risk Intelligence Platform")
st.markdown("**Predict hospital readmission risk for diabetic patients**")
st.divider()

st.sidebar.header("📋 Enter Patient Details")
age = st.sidebar.slider("Age Group (0=youngest, 9=oldest)", 0, 9, 5)
time_in_hospital = st.sidebar.slider("Days in Hospital", 1, 14, 3)
number_inpatient = st.sidebar.slider("Previous Inpatient Visits", 0, 10, 0)
number_diagnoses = st.sidebar.slider("Number of Diagnoses", 1, 16, 5)
num_medications = st.sidebar.slider("Number of Medications", 1, 81, 15)
num_lab_procedures = st.sidebar.slider("Lab Procedures", 1, 132, 45)

if st.sidebar.button("🔍 Predict Risk"):
    df_clean = pd.read_csv("diabetic_cleaned.csv")
    input_data = pd.DataFrame([df_clean.drop("readmitted", axis=1).iloc[0]])
    input_data["age"] = age
    input_data["time_in_hospital"] = time_in_hospital
    input_data["number_inpatient"] = number_inpatient
    input_data["number_diagnoses"] = number_diagnoses
    input_data["num_medications"] = num_medications
    input_data["num_lab_procedures"] = num_lab_procedures

    prob = model.predict_proba(input_data)[0][1]
    risk_percent = round(prob * 100, 1)

    st.subheader("🎯 Prediction Result")
    col1, col2 = st.columns(2)
    with col1:
        if risk_percent >= 40:
            st.error(f"⚠️ HIGH RISK: {risk_percent}%")
        elif risk_percent >= 20:
            st.warning(f"⚡ MEDIUM RISK: {risk_percent}%")
        else:
            st.success(f"✅ LOW RISK: {risk_percent}%")
    with col2:
        st.metric("Readmission Probability", f"{risk_percent}%")

    st.divider()
    st.subheader("📊 Key Risk Factors (from SHAP Analysis)")
    st.info("""
    - 🏥 **Previous inpatient visits** — strongest predictor
    - 🚪 **Discharge disposition** — where patient was sent after discharge
    - 🩺 **Primary diagnosis** — type of condition
    - 💊 **Diabetes medication** — treatment plan
    - ⏱️ **Time in hospital** — length of stay
    """)
else:
    st.info("👈 Enter patient details in the sidebar and click **Predict Risk**!")
    st.subheader("📊 About This Project")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Size", "101,766 patients")
    with col2:
        st.metric("Features Used", "44 clinical features")
    with col3:
        st.metric("Model", "XGBoost + SHAP")