import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Judul
st.set_page_config(page_title="Prediksi Pembelian Windows (di Toko Lisensi Official Kediri)", layout="centered")
st.title('Prediksi Pembelian Lisensi Windows Toko Lisensi Official Kediri')
st.subheader('Menggunakan Algoritma Random Forest & Support Vector Classifier (SVC)')
st.subheader('Kelompok 1')
st.subheader('Yogi Ario Pratama | 2313020004')
st.subheader('Shandy P | 2313020069')
st.subheader('Achmad Fardani | 232302008')

# baca model
MODEL_FILE = 'beli_windows.pkl' 
DATA_FILE = 'data.csv'

try:
    model_pipeline = joblib.load(MODEL_FILE)

    data = pd.read_csv(DATA_FILE, sep=',')
    pekerjaan_list = data['Pekerjaan'].unique().tolist()
    
except FileNotFoundError:
    st.error(f"File '{MODEL_FILE}' atau '{DATA_FILE}' tidak ditemukan. Pastikan Anda sudah menjalankan 'model_training.ipynb' dan file data ada.")
    st.stop()
except Exception as e:
    st.error(f"Error saat memuat model atau data: {e}")
    st.stop()


# proses prediksi
def make_prediction(input_df):
    """Melakukan prediksi menggunakan pipeline yang sudah dimuat."""
    prediction = model_pipeline.predict(input_df)
    return prediction[0]

# input dari user

st.markdown("Masukkan data pelanggan untuk memprediksi apakah mereka akan membeli lisensi Windows.")

with st.form("prediction_form"):
    
    st.markdown('##### Data Personal')
    usia = st.number_input('Usia', min_value=18, max_value=80, value=35, step=1)

    pekerjaan = st.selectbox('Pekerjaan', pekerjaan_list)
    
    penghasilan = st.number_input('Penghasilan Bulanan (Juta Rupiah)', min_value=0.0, max_value=50.0, value=7.5, step=0.1)

    st.markdown('##### Data Kebutuhan (Skala 0.0 - 1.0)')
    kemampuan_teknologi = st.number_input('Kemampuan Teknologi (Self-Rated)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    kebutuhan_bisnis = st.number_input('Kebutuhan Bisnis (Skala Penting)', min_value=0.0, max_value=1.0, value=0.6, step=0.01)

    submitted = st.form_submit_button("Prediksi Pembelian")

# urutan kolom
FEATURE_ORDER = ['Usia', 'Pekerjaan', 'Penghasilan_Bulanan_Juta', 'Kemampuan_Teknologi', 'Kebutuhan_Bisnis']

# --- E. hasil prediksi
if submitted:
    input_data_list = [usia, pekerjaan, penghasilan, kemampuan_teknologi, kebutuhan_bisnis]
    input_df = pd.DataFrame([input_data_list], columns=FEATURE_ORDER)
    
    result = make_prediction(input_df)
    
    st.divider()
    
    if result == 1:
        st.success('HASIL PREDIKSI: PELANGGAN DIPREDIKSI **AKAN MEMBELI** LISENSI WINDOWS!')
        st.balloons()
    else:
        st.error('HASIL PREDIKSI: PELANGGAN DIPREDIKSI **TIDAK AKAN MEMBELI** LISENSI WINDOWS.')
        
    # hasil
    model_name = model_pipeline.steps[1][0]

    st.info(f"Prediksi dilakukan menggunakan model terbaik ({model_name}).")
