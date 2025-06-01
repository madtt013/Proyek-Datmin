import streamlit as st
import pandas as pd
import joblib

st.title("🔮 Prediksi Stroke")

model = joblib.load('model.pkl')

def user_input():
    gender = st.selectbox("Jenis Kelamin", ['Male', 'Female'])
    age = st.slider("Usia", 1, 100)
    hypertension = st.selectbox("Hipertensi", [0, 1])
    heart_disease = st.selectbox("Penyakit Jantung", [0, 1])
    ever_married = st.selectbox("Pernah Menikah", ['Yes', 'No'])
    work_type = st.selectbox("Jenis Pekerjaan", ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
    residence_type = st.selectbox("Tipe Tempat Tinggal", ['Urban', 'Rural'])
    avg_glucose = st.slider("Glukosa Rata-rata", 50, 300)
    bmi = st.slider("BMI", 10.0, 50.0)
    smoking_status = st.selectbox("Status Merokok", ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

    data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': residence_type,
        'avg_glucose_level': avg_glucose,
        'bmi': bmi,
        'smoking_status': smoking_status
    }
    return pd.DataFrame([data])

input_df = user_input()
input_encoded = pd.get_dummies(input_df)
model_columns = joblib.load('model_columns.pkl')

for col in model_columns:
    if col not in input_encoded:
        input_encoded[col] = 0

input_encoded = input_encoded[model_columns]

if st.button("Prediksi"):
    prediction = model.predict(input_encoded)
    st.write(f"Hasil Prediksi: {'Stroke' if prediction[0]==1 else 'Tidak Stroke'}")
