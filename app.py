import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Memuat model dan objek preprocessing data
@st.cache_resource
def load_resources():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns_list = joblib.load("columns.pkl")
    return model, scaler, columns_list

try:
    model, scaler, columns_list = load_resources()
    resources_loaded = True
except Exception as e:
    st.error(f"Gagal memuat file model (best_model.pkl, scaler.pkl, atau columns.pkl). Pastikan file-file tersebut berada di repositori GitHub yang sama. Error: {e}")
    resources_loaded = False

st.title("Aplikasi Prediksi Customer Churn")
st.write("Aplikasi ini digunakan untuk memprediksi apakah pelanggan akan mempertahankan layanan (Retention) atau meninggalkan layanan (Churn) berdasarkan data historis.")

if resources_loaded:
    st.header("Metode 1: Unggah File CSV Data Pelanggan")
    uploaded_file = st.file_uploader("Pilih file CSV untuk prediksi massal", type=["csv"])

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.subheader("Sampel Data yang Diunggah")
            st.dataframe(df_input.head())

            # Proses pembersihan kolom yang tidak digunakan dalam pelatihan model
            kolom_tidak_dipakai = ['churn', 'customer_id', 'signup_date', 'last_purchase_date', 'coupon_code']
            df_clean = df_input.drop(columns=kolom_tidak_dipakai, errors='ignore')

            # One-Hot Encoding untuk fitur kategorikal
            df_encoded = pd.get_dummies(df_clean, drop_first=True)

            # Sinkronisasi kolom agar sesuai dengan urutan saat training
            df_encoded = df_encoded.reindex(columns=columns_list, fill_value=0)

            # Transformasi data dengan Scaler
            df_scaled = scaler.transform(df_encoded)

            # Prediksi menggunakan model hasil tuning
            predictions = model.predict(df_scaled)
            probabilities = model.predict_proba(df_scaled)[:, 1]

            # Menggabungkan hasil prediksi ke dataframe asli
            df_result = df_input.copy()
            df_result['Prediksi Churn'] = predictions
            df_result['Probabilitas Churn'] = probabilities
            df_result['Status Pelanggan'] = df_result['Prediksi Churn'].map({1: 'Churn', 0: 'Retain'})

            st.subheader("Hasil Analisis Prediksi")
            
            total_data = len(df_result)
            churn_count = int(df_result['Prediksi Churn'].sum())
            retain_count = total_data - churn_count
            
            st.write(f"Total baris data yang dianalisis: {total_data}")
            st.write(f"Pelanggan diprediksi Tetap (Retain): {retain_count} ({retain_count/total_data*100:.2f}%)")
            st.write(f"Pelanggan diprediksi Keluar (Churn): {churn_count} ({churn_count/total_data*100:.2f}%)")

            st.subheader("Data Hasil Prediksi")
            st.dataframe(df_result)

            # Fitur download hasil prediksi
            csv_data = df_result.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Unduh Hasil Prediksi (CSV)",
                data=csv_data,
                file_name="hasil_prediksi_churn.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file CSV: {e}")

    st.header("Metode 2: Input Manual Data Pelanggan Tunggal")
    st.write("Silakan masukkan nilai fitur di bawah ini untuk melihat hasil prediksi satu pelanggan khusus.")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Usia (Age)", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Jenis Kelamin (Gender)", ["Male", "Female"])
        total_spent = st.number_input("Total Pengeluaran (Total Spent)", min_value=0.0, value=500.0)
        avg_session_time = st.number_input("Rata-rata Waktu Sesi (Avg Session Time)", min_value=0.0, value=15.0)
        pages_per_session = st.number_input("Halaman per Sesi (Pages per Session)", min_value=0.0, value=4.0)

    with col2:
        satisfaction_score = st.slider("Skor Kepuasan (Satisfaction Score)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
        support_tickets = st.number_input("Jumlah Tiket Dukungan (Support Tickets)", min_value=0, value=1)
        marketing_spent_per_user = st.number_input("Biaya Pemasaran per Pengguna (Marketing Spent per User)", min_value=0.0, value=20.0)
        lifetime_value = st.number_input("Nilai Seumur Hidup Pelanggan (Lifetime Value)", min_value=0.0, value=1200.0)
        email_open_rate = st.slider("Rasio Membuka Email (Email Open Rate)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    if st.button("Prediksi Data Manual"):
        try:
            input_dict = {
                'age': [age],
                'gender': [gender],
                'total_spent': [total_spent],
                'avg_session_time': [avg_session_time],
                'pages_per_session': [pages_per_session],
                'satisfaction_score': [satisfaction_score],
                'support_tickets': [support_tickets],
                'marketing_spent_per_user': [marketing_spent_per_user],
                'lifetime_value': [lifetime_value],
                'email_open_rate': [email_open_rate]
            }
            
            df_manual = pd.DataFrame(input_dict)
            
            df_manual_encoded = pd.get_dummies(df_manual, drop_first=True)
            df_manual_encoded = df_manual_encoded.reindex(columns=columns_list, fill_value=0)
            
            df_manual_scaled = scaler.transform(df_manual_encoded)
            pred_manual = model.predict(df_manual_scaled)[0]
            prob_manual = model.predict_proba(df_manual_scaled)[0][1]
            
            st.subheader("Hasil Prediksi Manual")
            if pred_manual == 1:
                st.error(f"Hasil Prediksi: CHURN (Pelanggan kemungkinan besar akan keluar) dengan probabilitas {prob_manual*100:.2f}%")
            else:
                st.success(f"Hasil Prediksi: RETAIN (Pelanggan kemungkinan besar akan bertahan) dengan probabilitas {(1-prob_manual)*100:.2f}%")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan pada prediksi manual: {e}")
