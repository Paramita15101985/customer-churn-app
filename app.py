import streamlit as st
import joblib
import numpy as np

st.title("Customer Churn Prediction App")

# load model
model = joblib.load("model.joblib")

st.success("Model loaded successfully ✅")

# simple inputs (we will improve later)
tenure = st.number_input("Tenure")
monthly = st.number_input("Monthly Charges")
total = st.number_input("Total Charges")

if st.button("Predict"):
    input_data = np.array([[tenure, monthly, total]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer will CHURN ❌")
    else:
        st.success("Customer will NOT churn ✅")
