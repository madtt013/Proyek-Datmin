import pandas as pd

@st.cache_data
def load_data(path='data/stroke_data.csv'):
    df = pd.read_csv(path)
    return df
