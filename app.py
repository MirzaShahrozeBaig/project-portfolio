import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model, scaler, and feature columns
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl") # Load the column names

st.set_page_config(page_title="Telecom Churn Prediction", layout="wide")

st.title("📡 Telecom Customer Churn Prediction System")
st.markdown("Predict customer churn using Machine Learning")

st.sidebar.header("Customer Details")

# Input fields
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.25, 118.75, 70.0)
total_charges = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, 1500.0)

phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Conditional inputs based on Internet Service
if internet_service != "No":
    online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"])
    online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes"])
    device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes"])
    tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes"])
else:
    online_security = "No internet service"
    online_backup = "No internet service"
    device_protection = "No internet service"
    tech_support = "No internet service"
    streaming_tv = "No internet service"
    streaming_movies = "No internet service"

contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic_check", "Mailed_check", "Bank_transfer_(automatic)", "Credit_card_(automatic)"
])

def prepare_input_data(
    gender, senior_citizen, partner, dependents, tenure, monthly_charges, total_charges,
    phone_service, multiple_lines, internet_service, online_security, online_backup,
    device_protection, tech_support, streaming_tv, streaming_movies, contract,
    paperless_billing, payment_method, model_columns
):
    input_dict = {}

    # Numerical features
    input_dict['tenure'] = tenure
    input_dict['MonthlyCharges'] = monthly_charges
    input_dict['total charges'] = total_charges
    input_dict['AvgChargePerMonth'] = total_charges / (tenure + 1) if tenure > 0 else 0
    input_dict['Tenure_MonthlyCharges_Interaction'] = tenure * monthly_charges

    # Binary/Categorical features (based on model_columns)
    # Default all one-hot encoded columns to 0 first
    for col in model_columns:
        if col not in input_dict and ('_' in col or col == 'SeniorCitizen'): # Heuristic for one-hot encoded or special case
            input_dict[col] = 0

    # Set values based on user input
    input_dict['gender_male'] = 1 if gender == 'Male' else 0
    input_dict['SeniorCitizen'] = 1 if senior_citizen == 'Yes' else 0 # Original SeniorCitizen is 0/1 not one-hot encoded SeniorCitizen_1
    input_dict['Partner_Yes'] = 1 if partner == 'Yes' else 0
    input_dict['Dependents_Yes'] = 1 if dependents == 'Yes' else 0
    input_dict['PhoneService_Yes'] = 1 if phone_service == 'Yes' else 0

    if multiple_lines == 'No phone service':
        input_dict['MultipleLines_No phone service'] = 1
    elif multiple_lines == 'Yes':
        input_dict['MultipleLines_Yes'] = 1

    if internet_service == 'Fiber optic':
        input_dict['InternetService_Fiber optic'] = 1
    elif internet_service == 'No':
        input_dict['InternetService_No'] = 1

    if online_security == 'No internet service':
        input_dict['OnlineSecurity_No internet service'] = 1
    elif online_security == 'Yes':
        input_dict['OnlineSecurity_Yes'] = 1

    if online_backup == 'No internet service':
        input_dict['OnlineBackup_No internet service'] = 1
    elif online_backup == 'Yes':
        input_dict['OnlineBackup_Yes'] = 1

    if device_protection == 'No internet service':
        input_dict['DeviceProtection_No internet service'] = 1
    elif device_protection == 'Yes':
        input_dict['DeviceProtection_Yes'] = 1

    if tech_support == 'No internet service':
        input_dict['TechSupport_No internet service'] = 1
    elif tech_support == 'Yes':
        input_dict['TechSupport_Yes'] = 1

    if streaming_tv == 'No internet service':
        input_dict['StreamingTV_No internet service'] = 1
    elif streaming_tv == 'Yes':
        input_dict['StreamingTV_Yes'] = 1

    if streaming_movies == 'No internet service':
        input_dict['StreamingMovies_No internet service'] = 1
    elif streaming_movies == 'Yes':
        input_dict['StreamingMovies_Yes'] = 1

    if contract == 'One year':
        input_dict['Contract_One_year'] = 1
    elif contract == 'Two year':
        input_dict['Contract_Two_year'] = 1

    input_dict['PaperlessBilling_Yes'] = 1 if paperless_billing == 'Yes' else 0

    if payment_method == 'Credit_card_(automatic)':
        input_dict['PaymentMethod_Credit_card_(automatic)'] = 1
    elif payment_method == 'Electronic_check':
        input_dict['PaymentMethod_Electronic_check'] = 1
    elif payment_method == 'Mailed_check':
        input_dict['PaymentMethod_Mailed_check'] = 1

    # Create a DataFrame from the input and reindex to match the model's training columns
    input_df = pd.DataFrame([input_dict])
    # Ensure the order and presence of features match what the model expects
    input_aligned = input_df.reindex(columns=model_columns, fill_value=0)

    return input_aligned

if st.button("Predict"):
    input_data = prepare_input_data(
        gender, senior_citizen, partner, dependents, tenure, monthly_charges, total_charges,
        phone_service, multiple_lines, internet_service, online_security, online_backup,
        device_protection, tech_support, streaming_tv, streaming_movies, contract,
        paperless_billing, payment_method, model_columns
    )

    input_scaled = scaler.transform(input_data)

    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.error(f"❌ Customer WILL CHURN (Probability: {prob:.2%})")
    else:
        st.success(f"✅ Customer WILL STAY (Probability: {(1-prob):.2%})")
