import streamlit as st
import joblib
import numpy as np
import os

# -----------------------------
# App Title
# -----------------------------
st.title("Customer Churn Prediction App")

st.write("This is a Machine Learning web app for predicting customer churn.")

# -----------------------------
# Load Model Safely
# -----------------------------
MODEL_PATH = "model.joblib"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found in repository ❌")
    st.stop()

model = joblib.load(MODEL_PATH)

st.success("Model loaded successfully ✅")

# -----------------------------
# Input Section
# -----------------------------
st.subheader("Enter Customer Details")

tenure = st.number_input("Tenure (months)", min_value=0.0, step=1.0)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=1.0)
total_charges = st.number_input("Total Charges", min_value=0.0, step=1.0)

# -----------------------------
# Prediction Button (SAFE MODE)
# -----------------------------
if st.button("Predict Churn"):

    input_data = np.array([[tenure, monthly_charges, total_charges]])

    try:
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("Customer will CHURN ❌")
        else:
            st.success("Customer will NOT churn ✅")

    except Exception as e:
        st.error("Prediction error due to model mismatch.")
        st.write(str(e))
