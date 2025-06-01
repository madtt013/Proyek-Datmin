import pandas as pd
import streamlit as st

# Fungsi untuk memuat dataset
@st.cache_data
def load_data():
    DATA_PATH = "data/healthcare-dataset-stroke-data.csv"
    df = pd.read_csv(DATA_PATH)
    return df
