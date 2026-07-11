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
            st.error("Customer is likely to CHURN ❌")
        else:
            st.success("Customer is NOT likely to churn ✅")


        if probability is not None:
            st.info(
                f"Churn Probability: {probability*100:.2f}%"
            )


        # ---------------------------------
        # SHAP Explanation
        # ---------------------------------

        st.subheader("🔍 Model Interpretation (SHAP)")


        try:

            # If pipeline model
            if hasattr(model, "named_steps"):

                classifier = model.named_steps[
                    list(model.named_steps.keys())[-1]
                ]

                explainer = shap.TreeExplainer(
                    classifier
                )

            else:

                explainer = shap.TreeExplainer(
                    model
                )


            shap_values = explainer.shap_values(
                input_data
            )


            # Handle Random Forest binary output

            if isinstance(shap_values, list):

                values = shap_values[1][0]

                base_value = explainer.expected_value[1]


            elif len(shap_values.shape) == 3:

                values = shap_values[0,:,1]

                base_value = explainer.expected_value[1]


            else:

                values = shap_values[0]

                base_value = explainer.expected_value



            explanation = shap.Explanation(

                values=values,

                base_values=base_value,

                data=input_data[0],

                feature_names=[
                    "Tenure",
                    "Monthly Charges",
                    "Total Charges"
                ]
            )


            fig, ax = plt.subplots(
                figsize=(8,4)
            )


            shap.plots.waterfall(
                explanation,
                show=False
            )


            st.pyplot(fig)



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
    Project: Customer Churn Prediction
    """
)
