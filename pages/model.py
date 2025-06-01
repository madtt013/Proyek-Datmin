import streamlit as st
from utils.data_loader import load_data
from utils.model_utils import train_model
import json

st.title("📊 Pelatihan Model")

df = load_data()

if st.button("Train Model"):
    report = train_model(df)
    st.write("Model Performance:")
    st.json(report)
    
st.success("Model dilatih!")
st.subheader("Classification Report")
st.json(report)
