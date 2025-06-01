import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.model_utils import load_data

def run():
    st.title("Exploratory Data Analysis")
    df = load_data()

    st.subheader("Tampilkan Dataset")
    st.dataframe(df.head())

    st.subheader("Karakteristik Data")
    st.write(df.describe())

    st.subheader("Visualisasi Data")
    st.write("Distribusi Stroke:")
    fig, ax = plt.subplots()
    sns.countplot(x='stroke', data=df, ax=ax)
    st.pyplot(fig)

    st.write("Korelasi Antar Fitur:")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

