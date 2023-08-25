# Menambahkan CSS kustom untuk mengubah latar belakang
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import xgboost
import folium
from streamlit_folium import folium_static
st.set_page_config(
    page_title="Prediksi Level Kemacetan di 10 Titik Simpang Jakarta",
    page_icon="🚦",
    layout="centered",
    initial_sidebar_state="auto"
)
page_bg_img="""
<style>
[data-testid="stAppViewContainer"] > .main{
background-image: url("https://images.unsplash.com/photo-1588260369134-d64f66c5730b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1504&q=80");
background-size:100%;
background-position:top;
background-repeat:no-repeat;
background-attachment:local;
background-color: black !important;
}
[data-testid="stHeader"]{
background-color: rgba(0,0,0,0);
}
"""
st.markdown(page_bg_img,unsafe_allow_html=True)

model_regresi = pickle.load(open('prediksi_regresi.sav', 'rb'))
model_cluster = pickle.load(open('prediksi_cluster.sav', 'rb'))

st.title('🚦Prediksi Level Kemacetan di 10 Titik Simpang Jakarta')
df = pd.read_csv('simpangan_1.csv', delimiter=';')

# Create the selectbox for choosing 'simpang'
simpang_nama = st.selectbox('Pilih Simpang', df['simpang'])
simpang_data = df[df['simpang'] == simpang_nama].iloc[0]
base_duration = simpang_data['base_duration']
length = simpang_data['length']

st.write('You selected: ', simpang_nama)
st.write(f"Jarak yang dilalui: {length} m")
# # Buat peta awal dengan lokasi tengah
lokasi_asal,lokasi_tujuan =  simpang_data['origin'],simpang_data['destination']
lokasi_asal = lokasi_asal.strip('()').split(',')
latitude_asal,longitude_asal = float(lokasi_asal[0]),float(lokasi_asal[1])
lokasi_tujuan = lokasi_tujuan.strip('()').split(',')
latitude_tujuan,longitude_tujuan = float(lokasi_tujuan[0]),float(lokasi_tujuan[1])

m = folium.Map(location=[latitude_asal, longitude_asal], zoom_start=15)
# Tambahkan marker untuk titik koordinat pertama
folium.Marker([latitude_asal,longitude_asal], tooltip='Titik 1').add_to(m)
# Tambahkan marker untuk titik koordinat kedua
folium.Marker([latitude_tujuan,longitude_tujuan], tooltip='Titik 2').add_to(m)
folium_static(m)

# Load waktu data
df_waktu = pd.read_csv('waktu.csv')

# Select a time
selected_time = st.selectbox('Pilih Waktu', df_waktu['Waktu'])
st.write('You selected:', selected_time)

# Data preprocessing for time
selected_time = pd.to_datetime(selected_time)
jam = selected_time.hour
menit = selected_time.minute
Jam_radian = 2 * np.pi * jam / 24
Menit_radian = 2 * np.pi * menit / 60

# Calculate sin and cos values for time
Jam_sin = np.sin(Jam_radian)
Jam_cos = np.cos(Jam_radian)
Menit_sin = np.sin(Menit_radian)
Menit_cos = np.cos(Menit_radian)

# Data preprocessing for input
simpang = simpang_data['no_simpangan']
Kecepatan_normal = length / base_duration

input_data = [base_duration, length, simpang, Kecepatan_normal, Jam_radian, Menit_radian, Jam_sin, Jam_cos, Menit_sin, Menit_cos]
input_array = np.array(input_data).reshape(1, -1)

if st.button('Estimasi Waktu'):
    # Make predictions
    predict = model_regresi.predict(input_array)
    cluster = model_cluster.predict(input_array)

    st.write(f'Estimasi waktu yang dibutuhkan :  {"{:.2f}".format(predict[0])} detik')
    if cluster[0] == 0:
        st.write(f'Kondisi jalan dikatakan  <span style="color: red;background-color: black; display: inline; padding: 2px;">"Macet"</span>', unsafe_allow_html=True)
    elif cluster[0] == 3:
        st.write(f'Kondisi jalan dikatakan  <span style="color: orange;background-color: black; display: inline; padding: 2px;">"Sedikit Macet"</span>', unsafe_allow_html=True)
    elif cluster[0] == 1:
        st.write(f'Kondisi jalan dikatakan  <span style="color: yellow;background-color: black; display: inline; padding: 2px;">"Ramai Lancar"</span>', unsafe_allow_html=True)
    elif cluster[0] == 2:
        st.write(f'Kondisi jalan dikatakan  <span style="color: green;background-color: black; display: inline; padding: 2px;">"Lancar"</span>', unsafe_allow_html=True)
    else:
        st.write('Tidak ada informasi tentang kondisi lalu lintas')
