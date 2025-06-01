import streamlit as st
import pandas as pd
from utils.model_utils import load_model

def run():
    st.title("Formulir Prediksi")

    # Load model
    model = load_model()

    st.subheader("Masukkan Data")
    inputs = {
        "gender": st.selectbox("Gender", ["Male", "Female", "Other"]),
        "age": st.number_input("Age", value=0),
        "hypertension": st.radio("Hypertension", [0, 1]),
        "heart_disease": st.radio("Heart Disease", [0, 1]),
        "ever_married": st.selectbox("Ever Married", ["Yes", "No"]),
        "work_type": st.selectbox("Work Type", ["children", "Govt_job", "Never_worked", "Private", "Self-employed"]),
        "Residence_type": st.selectbox("Residence Type", ["Urban", "Rural"]),
        "avg_glucose_level": st.number_input("Average Glucose Level", value=0.0),
        "bmi": st.number_input("BMI", value=0.0),
        "smoking_status": st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])
    }

    if st.button("Prediksi"):
        df = pd.DataFrame([inputs])
        prediction = model.predict(df)
        st.write(f"Hasil Prediksi: {'Stroke' if prediction[0] == 1 else 'Tidak Stroke'}")

