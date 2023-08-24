import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Load the XGBoost models
model_regresi = pickle.load(open('prediksi_regresi.sav', 'rb'))
model_cluster = pickle.load(open('prediksi_cluster.sav', 'rb'))

st.title('Prediksi Level Kemacetan')
df = pd.read_csv('simpangan_1.csv', delimiter=';')

# Create the selectbox for choosing 'simpang'
simpang_nama = st.selectbox('Pilih Simpang', df['simpang'])
simpang_data = df[df['simpang'] == simpang_nama].iloc[0]
base_duration = simpang_data['base_duration']
length = simpang_data['length']

st.write('You selected:', simpang_nama)
st.write(f"Waktu yang ditempuh ketika normal: {base_duration} detik")
st.write(f"Jarak yang dilalui: {length} m")

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

    st.write('Estimasi waktu yang dibutuhkan : ', predict[0])
    st.write('Cluster : ', cluster[0])
