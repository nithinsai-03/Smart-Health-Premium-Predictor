# main.py
import streamlit as st
from prediction_helper import predict

st.set_page_config(page_title="Health Insurance Predictor", layout="wide")
st.title("💼 Health Insurance Cost Predictor")

# Dropdown options
categorical_options = {
    'Gender': ['Male', 'Female'],
    'Marital Status': ['Unmarried', 'Married'],
    'BMI Category': ['Normal', 'Obesity', 'Overweight', 'Underweight'],
    'Smoking Status': ['No Smoking', 'Regular', 'Occasional'],
    'Employment Status': ['Salaried', 'Self-Employed', 'Freelancer'],
    'Region': ['Northwest', 'Southeast', 'Northeast', 'Southwest'],
    'Medical History': [
        'No Disease', 'Diabetes', 'High blood pressure', 'Diabetes & High blood pressure',
        'Thyroid', 'Heart disease', 'High blood pressure & Heart disease',
        'Diabetes & Thyroid', 'Diabetes & Heart disease'
    ],
    'Insurance Plan': ['Bronze', 'Silver', 'Gold']
}

# Input layout
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input('Age', 18, 100, 30)
with col2:
    number_of_dependants = st.number_input('Number of Dependants', 0, 20, 0)
with col3:
    income_lakhs = st.number_input('Income in Lakhs', 0, 200, 10)

col4, col5, col6 = st.columns(3)
with col4:
    genetical_risk = st.number_input('Genetical Risk', 0, 5, 0)
with col5:
    insurance_plan = st.selectbox('Insurance Plan', categorical_options['Insurance Plan'])
with col6:
    employment_status = st.selectbox('Employment Status', categorical_options['Employment Status'])

col7, col8, col9 = st.columns(3)
with col7:
    gender = st.selectbox('Gender', categorical_options['Gender'])
with col8:
    marital_status = st.selectbox('Marital Status', categorical_options['Marital Status'])
with col9:
    bmi_category = st.selectbox('BMI Category', categorical_options['BMI Category'])

col10, col11, col12 = st.columns(3)
with col10:
    smoking_status = st.selectbox('Smoking Status', categorical_options['Smoking Status'])
with col11:
    region = st.selectbox('Region', categorical_options['Region'])
with col12:
    medical_history = st.selectbox('Medical History', categorical_options['Medical History'])

# Create input dictionary
input_dict = {
    'Age': age,
    'Number of Dependants': number_of_dependants,
    'Income in Lakhs': income_lakhs,
    'Genetical Risk': genetical_risk,
    'Insurance Plan': insurance_plan,
    'Employment Status': employment_status,
    'Gender': gender,
    'Marital Status': marital_status,
    'BMI Category': bmi_category,
    'Smoking Status': smoking_status,
    'Region': region,
    'Medical History': medical_history
}

# Predict button
if st.button("Predict"):
    premium = predict(input_dict)
    st.success(f"💰 Predicted Health Insurance Cost: ₹ {premium}")
