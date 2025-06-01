import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.data_loader import stroke_data.csv

st.title("🧠 Exploratory Data Analysis (EDA)")

df = load_data()

st.subheader("Dataset")
st.dataframe(df.head())

st.subheader("Informasi Data")
st.write(df.describe())

st.subheader("Distribusi Target")
fig, ax = plt.subplots()
sns.countplot(data=df, x='stroke', ax=ax)
st.pyplot(fig)

st.subheader("Heatmap Korelasi")
fig, ax = plt.subplots()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)
