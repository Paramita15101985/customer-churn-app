import streamlit as st
import joblib
import numpy as np
import os
import shap
import matplotlib.pyplot as plt

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# ---------------------------------
# Title
# ---------------------------------
st.title("📊 Customer Churn Prediction App")

st.write(
    "This Machine Learning web application predicts whether a customer is likely to churn."
)

# ---------------------------------
# Load Model
# ---------------------------------

MODEL_PATH = "churn_model.joblib"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found in repository ❌")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    st.success("Model loaded successfully ✅")

except EOFError:
    st.error(
        "Model file is corrupted or empty ❌\n\n"
        "Please retrain the model and upload a new churn_model.joblib file."
    )
    st.stop()

except Exception as e:
    st.error("Unable to load model ❌")
    st.write(e)
    st.stop()


# ---------------------------------
# Customer Input
# ---------------------------------

st.subheader("Enter Customer Details")

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges ($)",
    min_value=0.0,
    value=50.0
)

total_charges = st.number_input(
    "Total Charges ($)",
    min_value=0.0,
    value=600.0
)


# ---------------------------------
# Prediction
# ---------------------------------

if st.button("Predict Churn"):

    input_data = np.array(
        [[
            tenure,
            monthly_charges,
            total_charges
        ]]
    )

    try:

        prediction = model.predict(input_data)

        probability = None

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1]


        if prediction[0] == 1:

            st.error(
                "Customer is likely to CHURN ❌"
            )

        else:

            st.success(
                "Customer is NOT likely to churn ✅"
            )


        if probability is not None:

            st.info(
                f"Churn Probability: {probability*100:.2f}%"
            )


        # ---------------------------------
        # SHAP Explanation
        # ---------------------------------

        st.subheader("🔍 Model Interpretation (SHAP)")


        try:

            explainer = shap.Explainer(model)

            shap_values = explainer(input_data)


            feature_names = [
                "Tenure",
                "Monthly Charges",
                "Total Charges"
            ]


            fig = plt.figure(figsize=(8,4))


            shap.plots.waterfall(
                shap_values[0],
                show=False
            )


            st.pyplot(
                plt.gcf()
            )


        except Exception as shap_error:

            st.warning(
                "SHAP explanation unavailable for this model."
            )

            st.write(shap_error)



    except Exception as e:

        st.error(
            "Prediction error ❌"
        )

        st.write(e)



# ---------------------------------
# Footer
# ---------------------------------

st.markdown(
    """
    ---
    Developed using Machine Learning  
    Algorithm: Random Forest / ML Pipeline  
    Project: Customer Churn Prediction
    """
)
