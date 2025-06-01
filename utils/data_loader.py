import pandas as pd

@st.cache
def load_data():
    return pd.read_csv("data/stroke_data.csv")
