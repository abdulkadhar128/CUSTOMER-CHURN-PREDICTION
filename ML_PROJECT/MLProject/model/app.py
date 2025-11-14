import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# Load the best model
# -----------------------------
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

best_model = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Prediction App", page_icon="🤖")
st.title("🤖 Customer Prediction using Best Model")
st.write("This app predicts the target based on the trained model (Logistic Regression).")

# Input fields (example – replace with your dataset features)
st.subheader("Enter Input Values")

# ❗️ Change feature inputs according to your dataset
age = st.number_input("Age", min_value=18, max_value=100, step=1)
salary = st.number_input("Monthly Salary", min_value=0, step=1000)
gender = st.selectbox("Gender", ["Male", "Female"])
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1)

# Convert categorical input to numeric (modify based on your preprocessing)
gender = 1 if gender == "Male" else 0

if st.button("Predict"):
    try:
        # Arrange data into dataframe or array according to your model input shape
        input_data = np.array([[age, salary, gender, credit_score]])

        prediction = best_model.predict(input_data)
        prediction_prob = best_model.predict_proba(input_data)

        st.success(f"✅ Prediction: **{prediction[0]}**")
        st.write(f"📊 Probability of prediction: {prediction_prob}")

    except Exception as e:
        st.error(f"❌ Error: {e}")


st.markdown("---")
st.caption("Developed by Abdul Kadhar | Powered by GPT-5 🚀")
