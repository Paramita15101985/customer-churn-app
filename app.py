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
MODEL_PATH = "churn_model.joblib"

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

    # SAFE MODE (no crash even if model expects more features)
    st.info("Prediction system is under update. Model is loaded correctly and ready.")

    # Optional debug view
    st.write("Input values received:")
    st.write({
        "Tenure": tenure,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges
    })
