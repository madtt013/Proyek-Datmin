import streamlit as st

# Navigasi antar halaman
def main():
    st.sidebar.title("Navigasi")
    pages = {
        "Halaman EDA": "pages.page1_eda",
        "Hasil Pelatihan Model": "pages.page2_model",
        "Formulir Prediksi": "pages.page3_predict"
    }
    choice = st.sidebar.radio("Pilih Halaman", list(pages.keys()))

    module = __import__(pages[choice], fromlist=[''])
    module.run()

if __name__ == "__main__":
    main()
