import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model, scaler, and feature columns
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl") # Load the column names

st.set_page_config(page_title="Telecom Churn Prediction")

st.title("📡 Telecom Customer Churn Prediction System")
st.markdown("Predict customer churn using Machine Learning")

st.sidebar.header("Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

tenure = st.sidebar.slider("Tenure", 0, 72, 12)
monthly = st.sidebar.slider("Monthly Charges", 20, 150, 70)
total = st.sidebar.slider("Total Charges", 0, 9000, 1500) # This will be the 'total charges' float

internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

def prepare_input_data(gender, senior, partner, dependents, tenure, monthly, total, internet, contract, payment):
    input_dict = {
        'gender_male': 1 if gender == 'Male' else 0,
        'SeniorCitizen_1': 1 if senior == 'Yes' else 0,
        'Partner_Yes': 1 if partner == 'Yes' else 0,
        'Dependents_Yes': 1 if dependents == 'Yes' else 0,
        'tenure': tenure,
        'PhoneService_Yes': 1, # Assuming phone service is generally 'Yes' for simplicity in app
        'MultipleLines_No phone service': 0, # Assuming not 'No phone service'
        'MultipleLines_Yes': 0, # Assuming 'No' for simplicity
        'InternetService_Fiber optic': 1 if internet == 'Fiber optic' else 0,
        'InternetService_No': 1 if internet == 'No' else 0,
        'OnlineSecurity_No internet service': 1 if internet == 'No' else 0,
        'OnlineSecurity_Yes': 0, # Assuming 'No' for simplicity
        'OnlineBackup_No internet service': 1 if internet == 'No' else 0,
        'OnlineBackup_Yes': 0, # Assuming 'No' for simplicity
        'DeviceProtection_No internet service': 1 if internet == 'No' else 0,
        'DeviceProtection_Yes': 0, # Assuming 'No' for simplicity
        'TechSupport_No internet service': 1 if internet == 'No' else 0,
        'TechSupport_Yes': 0, # Assuming 'No' for simplicity
        'StreamingTV_No internet service': 1 if internet == 'No' else 0,
        'StreamingTV_Yes': 0, # Assuming 'No' for simplicity
        'StreamingMovies_No internet service': 1 if internet == 'No' else 0,
        'StreamingMovies_Yes': 0, # Assuming 'No' for simplicity
        'Contract_One_year': 1 if contract == 'One year' else 0,
        'Contract_Two_year': 1 if contract == 'Two year' else 0,
        'PaperlessBilling_Yes': 0, # Assuming 'No' for simplicity
        'PaymentMethod_Credit_card_(automatic)': 1 if payment == 'Credit card (automatic)' else 0,
        'PaymentMethod_Electronic_check': 1 if payment == 'Electronic check' else 0,
        'PaymentMethod_Mailed_check': 1 if payment == 'Mailed check' else 0,
        'MonthlyCharges': monthly,
        'total charges': total, # This corresponds to the 'total charges' float column
        'AvgChargePerMonth': total / (tenure + 1) if tenure > 0 else 0,
        'Tenure_MonthlyCharges_Interaction': tenure * monthly
    }

    # Create a DataFrame from the input and reindex to match the model's training columns
    input_df = pd.DataFrame([input_dict])
    # Align columns, filling missing with 0 (for categorical features not selected)
    input_aligned = input_df.reindex(columns=model_columns, fill_value=0)

    return input_aligned

if st.button("Predict"):    
    input_data = prepare_input_data(
        gender, senior, partner, dependents, tenure, monthly, total,
        internet, contract, payment
    )

    input_scaled = scaler.transform(input_data)

    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.error(f"❌ Customer WILL CHURN (Probability: {prob:.2%})")
    else:
        st.success(f"✅ Customer WILL STAY (Probability: {(1-prob):.2%})")
