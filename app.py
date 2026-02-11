import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model & scaler
import joblib

model = joblib.load('churn_model.joblib')
scaler = joblib.load('scaler.joblib')
# Load training column structure
columns = pickle.load(open('df.pkl', 'rb')).columns

st.set_page_config(page_title="Telecom Churn Predictor", layout="wide")

st.title("📡 Telecom Customer Churn Prediction System")
st.write("Predict customer churn using Machine Learning")

# ---------------- USER INPUTS ----------------
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", ["Yes", "No"])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
MonthlyCharges = st.slider("Monthly Charges", 0, 150, 70)
TotalCharges = st.slider("Total Charges", 0, 10000, 1500)
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaymentMethod = st.selectbox("Payment Method", 
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# ---------------- ENCODING ----------------
gender = 1 if gender == "Male" else 0
SeniorCitizen = 1 if SeniorCitizen == "Yes" else 0
Partner = 1 if Partner == "Yes" else 0
Dependents = 1 if Dependents == "Yes" else 0

# Base input dictionary
input_dict = {
    'gender': gender,
    'SeniorCitizen': SeniorCitizen,
    'Partner': Partner,
    'Dependents': Dependents,
    'tenure': tenure,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges
}

# One-hot style categorical handling
for col in columns:
    if col.startswith("InternetService_"):
        input_dict[col] = 1 if col == f"InternetService_{InternetService}" else 0
    if col.startswith("Contract_"):
        input_dict[col] = 1 if col == f"Contract_{Contract}" else 0
    if col.startswith("PaymentMethod_"):
        input_dict[col] = 1 if col == f"PaymentMethod_{PaymentMethod}" else 0

# Create dataframe in correct order
input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=columns, fill_value=0)

# Scale
input_scaled = scaler.transform(input_df)

# ---------------- PREDICT ----------------
if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠ Customer likely to CHURN — Probability: {prob:.2%}")
    else:
        st.success(f"✅ Customer likely to STAY — Probability: {1-prob:.2%}")
