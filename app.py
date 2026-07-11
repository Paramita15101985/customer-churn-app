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

import shap
import matplotlib.pyplot as plt

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


explainer = shap.TreeExplainer(model)

sample = np.array([[tenure, monthly, total]])

shap_values = explainer.shap_values(sample)

st.subheader("Model Interpretation (SHAP)")

fig, ax = plt.subplots(figsize=(8,3))

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[1][0],
        base_values=explainer.expected_value[1],
        data=sample[0],
        feature_names=[
            "Tenure",
            "Monthly Charges",
            "Total Charges"
        ]
    ),
    show=False
)

st.pyplot(fig)
