import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------
# Load Models and Scalers
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_young_path = os.path.join(BASE_DIR, "model_young.joblib")
model_rest_path = os.path.join(BASE_DIR, "model_rest.joblib")
scaler_young_path = os.path.join(BASE_DIR, "scaler_with_cols_young.joblib")
scaler_rest_path = os.path.join(BASE_DIR, "scalar_with_cols_rest.joblib")

try:
    model_young = joblib.load(model_young_path)
    model_rest = joblib.load(model_rest_path)
    scaler_young = joblib.load(scaler_young_path)
    scaler_rest = joblib.load(scaler_rest_path)
    st.sidebar.success("✅ Models and scalers loaded successfully!")
except Exception as e:
    st.sidebar.error(f"⚠️ Error loading model/scaler files: {e}")
    st.stop()  # Stop the app if files not loaded

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title=" Health Insurance Cost Predictor",
    page_icon="💰",
    layout="centered"
)

st.title("💊 Health Insurance Cost Predictor")
st.markdown("Fill in your details to get an estimated health insurance premium.")

st.divider()

# ---------------------------------------------------------
# Input Options
# ---------------------------------------------------------
gender_options = ['Male', 'Female']
region_options = ['Northwest', 'Southeast', 'Northeast', 'Southwest']
marital_options = ['Unmarried', 'Married']
bmi_options = ['Normal', 'Obesity', 'Overweight', 'Underweight']
smoking_options = ['No Smoking', 'Regular', 'Occasional']
employment_options = ['Salaried', 'Self-Employed', 'Freelancer']
income_options = ['<10L', '10L - 25L', '25L - 40L', '> 40L']
medical_options = [
    'No Disease', 'Diabetes', 'High blood pressure', 'Heart disease',
    'Diabetes & High blood pressure', 'High blood pressure & Heart disease',
    'Diabetes & Thyroid', 'Diabetes & Heart disease', 'Thyroid'
]
insurance_plan_options = ['Bronze', 'Silver', 'Gold']

# ---------------------------------------------------------
# User Input Form
# ---------------------------------------------------------
with st.form("predict_form"):
    st.subheader("Enter Your Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", gender_options)
        region = st.selectbox("Region", region_options)
        marital_status = st.selectbox("Marital Status", marital_options)
        bmi_category = st.selectbox("BMI Category", bmi_options)
        smoking_status = st.selectbox("Smoking Status", smoking_options)
        age = st.slider("Age", 18, 80, 30)

    with col2:
        employment_status = st.selectbox("Employment Status", employment_options)
        income_level = st.selectbox("Income Level", income_options)
        medical_condition = st.selectbox("Medical Condition", medical_options)
        insurance_plan = st.selectbox("Insurance Plan", insurance_plan_options)
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)

    submitted = st.form_submit_button("🔍 Predict Premium")

# ---------------------------------------------------------
# Map categorical inputs to numeric (must match training)
# ---------------------------------------------------------
gender_map = {'Male':0, 'Female':1}
region_map = {'Northwest':0, 'Southeast':1, 'Northeast':2, 'Southwest':3}
marital_map = {'Unmarried':0, 'Married':1}
bmi_map = {'Normal':0, 'Obesity':1, 'Overweight':2, 'Underweight':3}
smoking_map = {'No Smoking':0, 'Regular':1, 'Occasional':2}
employment_map = {'Salaried':0, 'Self-Employed':1, 'Freelancer':2}
income_map = {'<10L':0, '10L - 25L':1, '25L - 40L':2, '> 40L':3}
medical_map = {
    'No Disease':0, 'Diabetes':1, 'High blood pressure':2, 'Heart disease':3,
    'Diabetes & High blood pressure':4, 'High blood pressure & Heart disease':5,
    'Diabetes & Thyroid':6, 'Diabetes & Heart disease':7, 'Thyroid':8
}
plan_map = {'Bronze':0, 'Silver':1, 'Gold':2}

# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------
if submitted:
    # Convert to numeric DataFrame
    input_numeric = pd.DataFrame({
        'Gender':[gender_map[gender]],
        'Region':[region_map[region]],
        'Marital Status':[marital_map[marital_status]],
        'BMI Category':[bmi_map[bmi_category]],
        'Smoking Status':[smoking_map[smoking_status]],
        'Employment Status':[employment_map[employment_status]],
        'Income Level':[income_map[income_level]],
        'Medical Condition':[medical_map[medical_condition]],
        'Insurance Plan':[plan_map[insurance_plan]],
        'Age':[age],
        'Dependents':[dependents]
    })

    st.subheader("📋 Your Input Summary")
    st.json({
        "Gender":gender,
        "Region":region,
        "Marital Status":marital_status,
        "BMI Category":bmi_category,
        "Smoking Status":smoking_status,
        "Employment Status":employment_status,
        "Income Level":income_level,
        "Medical Condition":medical_condition,
        "Insurance Plan":insurance_plan,
        "Age":age,
        "Dependents":dependents
    })

    try:
        # Choose model and scaler
        if age <= 35:
            model = model_young
            scaler = scaler_young
            st.info("Using **Young Group Model** 🧒")
        else:
            model = model_rest
            scaler = scaler_rest
            st.info("Using **Rest Group Model** 🧓")

        # Scale and predict
        input_scaled = scaler.transform(input_numeric)
        prediction = model.predict(input_scaled)
        st.success(f"💰 Estimated Annual Premium: ₹ {prediction[0]:,.2f}")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()
st.markdown("""
### ℹ️ About This App
AI-powered Health Insurance Premium Predictor.  
Automatically selects model/scaler based on your age and applies preprocessing to your inputs.

**Developed by:** Nithin Sai
""")
