import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline

# Load the preprocessor and the model
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('best_heart_disease_model.pkl')

# Define the input fields
st.title("Heart Disease Prediction")
st.write("Enter the patient details to predict the likelihood of heart disease.")

age = st.number_input("Age", min_value=1, max_value=120, value=25)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=300, value=120)
chol = st.number_input("Serum Cholesterol (chol)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=50, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])

# Make prediction
if st.button("Predict"):
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })

    # Preprocess the input data
    input_data_preprocessed = preprocessor.transform(input_data)

    # Predict using the loaded model
    prediction = model.predict(input_data_preprocessed)
    prediction_proba = model.predict_proba(input_data_preprocessed)

    if prediction[0] == 1:
        st.error(f"The patient is likely to have heart disease. (Probability: {prediction_proba[0][1]:.2f})")
    else:
        st.success(f"The patient is unlikely to have heart disease. (Probability: {prediction_proba[0][0]:.2f})")