import streamlit as st
from utils.model_utils import train_model

def run():
    st.title("Hasil Pelatihan Model")
    accuracy, report = train_model()

    st.subheader("Akurasi Model")
    st.write(f"Akurasi: {accuracy:.2f}")

    st.subheader("Laporan Klasifikasi")
    st.text(report)

