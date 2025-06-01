import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('stroke_data.csv')
    return data

df = load_data()

# Preprocessing
def preprocess_data(data):
    df = data.copy()
    df = df.dropna()
    df['gender'] = df['gender'].replace('Other', 'Male')
    le = LabelEncoder()
    for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
        df[col] = le.fit_transform(df[col])
    X = df.drop(columns=['id', 'stroke'])
    y = df['stroke']
    return X, y

# Sidebar Navigation
page = st.sidebar.selectbox("Pilih Halaman", ["EDA", "Model", "Prediksi"])

# =======================
# 1. Halaman EDA
# =======================
if page == "EDA":
    st.title("📊 Exploratory Data Analysis - Stroke Prediction")
    st.write("Berikut adalah tampilan awal dari dataset:")
    st.dataframe(df.head())

    st.subheader("Distribusi Target (Stroke)")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='stroke', ax=ax)
    st.pyplot(fig)

    st.subheader("Distribusi Usia")
    fig, ax = plt.subplots()
    sns.histplot(df['age'], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Stroke berdasarkan Gender")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='gender', hue='stroke', ax=ax)
    st.pyplot(fig)

# =======================
# 2. Halaman Model
# =======================
elif page == "Model":
    st.title("🤖 Training Model - Stroke Prediction")

    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    st.write(f"🎯 Akurasi model: **{accuracy:.2%}**")

    # Simpan model
    with open("stroke_model.pkl", "wb") as f:
        pickle.dump(model, f)
    st.success("Model telah dilatih dan disimpan sebagai `stroke_model.pkl`.")

# =======================
# 3. Halaman Prediksi
# =======================
elif page == "Prediksi":
    st.title("🩺 Formulir Prediksi Stroke")

    with open("stroke_model.pkl", "rb") as f:
        model = pickle.load(f)

    st.write("Masukkan data pasien:")

    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Umur", 0, 100, 30)
    hypertension = st.selectbox("Hipertensi", [0, 1])
    heart_disease = st.selectbox("Penyakit Jantung", [0, 1])
    ever_married = st.selectbox("Pernah Menikah", ["Yes", "No"])
    work_type = st.selectbox("Jenis Pekerjaan", ["Private", "Self-employed", "Govt_job", "children"])
    Residence_type = st.selectbox("Tempat Tinggal", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Rata-rata Glukosa", 50.0, 300.0, 100.0)
    bmi = st.number_input("BMI", 10.0, 50.0, 20.0)
    smoking_status = st.selectbox("Status Merokok", ["formerly smoked", "never smoked", "smokes"])

    input_data = pd.DataFrame({
        'gender': [0 if gender == "Male" else 1],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [1 if ever_married == "Yes" else 0],
        'work_type': [0 if work_type == "Govt_job" else 1 if work_type == "Private" else 2 if work_type == "Self-employed" else 3],
        'Residence_type': [0 if Residence_type == "Urban" else 1],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [0 if smoking_status == "formerly smoked" else 1 if smoking_status == "never smoked" else 2]
    })

    if st.button("Prediksi Stroke"):
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.error("⚠️ Pasien berisiko terkena stroke.")
        else:
            st.success("✅ Pasien tidak berisiko stroke.")

