# prediction_helper.py
import pandas as pd
import joblib

# Load models and scalers (from the same folder)
model_young = joblib.load("model_young.joblib")
model_rest = joblib.load("model_rest.joblib")
scaler_young = joblib.load("scaler_with_cols_young.joblib")
scaler_rest = joblib.load("scalar_with_cols_rest.joblib")  # note actual file name

def calculate_normalized_risk(medical_history):
    """Calculate normalized risk score based on medical history."""
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    diseases = medical_history.lower().split(" & ")
    total_score = sum(risk_scores.get(d, 0) for d in diseases)
    max_score = 14
    min_score = 0
    return (total_score - min_score) / (max_score - min_score)

def preprocess_input(input_dict):
    """Preprocess input dictionary into model-ready DataFrame."""
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight',
        'smoking_status_Occasional', 'smoking_status_Regular',
        'employment_status_Salaried', 'employment_status_Self-Employed'
    ]
    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}

    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    # Encode categorical variables
    if input_dict['Gender'] == 'Male':
        df['gender_Male'] = 1
    if input_dict['Region'] == 'Northwest':
        df['region_Northwest'] = 1
    elif input_dict['Region'] == 'Southeast':
        df['region_Southeast'] = 1
    elif input_dict['Region'] == 'Southwest':
        df['region_Southwest'] = 1
    if input_dict['Marital Status'] == 'Unmarried':
        df['marital_status_Unmarried'] = 1
    if input_dict['BMI Category'] == 'Obesity':
        df['bmi_category_Obesity'] = 1
    elif input_dict['BMI Category'] == 'Overweight':
        df['bmi_category_Overweight'] = 1
    elif input_dict['BMI Category'] == 'Underweight':
        df['bmi_category_Underweight'] = 1
    if input_dict['Smoking Status'] == 'Occasional':
        df['smoking_status_Occasional'] = 1
    elif input_dict['Smoking Status'] == 'Regular':
        df['smoking_status_Regular'] = 1
    if input_dict['Employment Status'] == 'Salaried':
        df['employment_status_Salaried'] = 1
    elif input_dict['Employment Status'] == 'Self-Employed':
        df['employment_status_Self-Employed'] = 1

    # Assign numerical values
    df['insurance_plan'] = insurance_plan_encoding.get(input_dict['Insurance Plan'], 1)
    df['age'] = input_dict['Age']
    df['number_of_dependants'] = input_dict['Number of Dependants']
    df['income_lakhs'] = input_dict['Income in Lakhs']
    df['genetical_risk'] = input_dict['Genetical Risk']
    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])

    # Scaling numerical columns
    if input_dict['Age'] <= 25:
        scaler_obj = scaler_young
    else:
        scaler_obj = scaler_rest

    scaler = scaler_obj['scaler']
    cols_to_scale = scaler_obj['cols_to_scale']
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df

def predict(input_dict):
    """Return predicted insurance premium based on input dictionary."""
    df = preprocess_input(input_dict)
    if input_dict['Age'] <= 25:
        prediction = model_young.predict(df)
    else:
        prediction = model_rest.predict(df)
    return int(prediction[0])
