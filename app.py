import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Konfigurasi halaman agar tampilan lebih luas dan profesional
st.set_page_config(
    page_title="Churn Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Memuat model dan objek preprocessing
@st.cache_resource
def load_resources():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns_list = joblib.load("columns.pkl")
    return model, scaler, columns_list

# Load resources
try:
    model, scaler, columns_list = load_resources()
    resources_loaded = True
except Exception as e:
    st.error(f"Gagal memuat file sistem. Pastikan best_model.pkl, scaler.pkl, dan columns.pkl ada di GitHub.")
    resources_loaded = False

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.title("Main Menu")
    navigation = st.radio(
        "Pilih Menu:",
        ["Dashboard", "Analisis Prediksi Massal", "Prediksi Manual", "Informasi Model"]
    )
    st.divider()
    st.info("Aplikasi ini menggunakan algoritma Machine Learning untuk memprediksi risiko perpindahan pelanggan (Churn).")

if resources_loaded:
    
    # HALAMAN 1: DASHBOARD UTAMA
    if navigation == "Dashboard":
        st.title("Dashboard Prediksi Customer Churn")
        st.write("Selamat datang di sistem pendukung keputusan manajemen pelanggan. Gunakan menu di samping untuk mulai melakukan prediksi.")
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.subheader("Tujuan Aplikasi")
            st.write("""
            1. Mengidentifikasi pelanggan yang berisiko berhenti menggunakan layanan.
            2. Memberikan wawasan data melalui visualisasi.
            3. Membantu tim pemasaran dalam menentukan strategi retensi.
            """)
        with col_info2:
            st.subheader("Parameter Utama")
            st.write("- Kepuasan Pelanggan\n- Pengeluaran Total\n- Waktu Sesi Aplikasi\n- Jumlah Tiket Dukungan")

    # HALAMAN 2: ANALISIS PREDIKSI MASSAL (CSV)
    elif navigation == "Analisis Prediksi Massal":
        st.title("Analisis Prediksi Massal via CSV")
        uploaded_file = st.file_uploader("Unggah berkas dataset CSV Anda di sini", type=["csv"])

        if uploaded_file is not None:
            df_input = pd.read_csv(uploaded_file)
            
            # --- PROSES PREDIKSI ---
            kolom_tidak_dipakai = ['churn', 'customer_id', 'signup_date', 'last_purchase_date', 'coupon_code']
            df_clean = df_input.drop(columns=kolom_tidak_dipakai, errors='ignore')
            df_encoded = pd.get_dummies(df_clean, drop_first=True)
            df_encoded = df_encoded.reindex(columns=columns_list, fill_value=0)
            df_scaled = scaler.transform(df_encoded)
            
            predictions = model.predict(df_scaled)
            df_input['Status'] = np.where(predictions == 1, 'Churn', 'Retain')

            # --- VISUALISASI HASIL (PIE/DONUT CHART) ---
            st.divider()
            col_metric1, col_metric2 = st.columns([1, 1])
            
            total_data = len(df_input)
            churn_count = int((df_input['Status'] == 'Churn').sum())
            retain_count = total_data - churn_count

            with col_metric1:
                st.subheader("Ringkasan Statistik")
                st.metric("Total Pelanggan", f"{total_data} orang")
                st.metric("Prediksi Churn", f"{churn_count} orang", delta=f"{(churn_count/total_data)*100:.1f}%", delta_color="inverse")
                st.metric("Prediksi Retain", f"{retain_count} orang", delta=f"{(retain_count/total_data)*100:.1f}%")

            with col_metric2:
                st.subheader("Proporsi Status Pelanggan")
                # Membuat Donut Chart
                fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
                colors = ['#2ecc71', '#e74c3c'] # Hijau untuk Retain, Merah untuk Churn
                labels = ['Retain', 'Churn']
                sizes = [retain_count, churn_count]
                
                ax_pie.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, 
                           pctdistance=0.85, explode=(0.05, 0.05))
                
                # Menggambar lingkaran di tengah untuk efek Donut
                centre_circle = plt.Circle((0,0), 0.70, fc='white')
                fig_pie.gca().add_artist(centre_circle)
                
                ax_pie.axis('equal')  
                plt.tight_layout()
                st.pyplot(fig_pie)

            st.divider()
            st.subheader("Tabel Detail Prediksi")
            st.dataframe(df_input, use_container_width=True)
            
            csv = df_input.to_csv(index=False).encode('utf-8')
            st.download_button("Unduh Hasil Prediksi Lengkap", csv, "hasil_prediksi.csv", "text/csv")

    # HALAMAN 3: PREDIKSI MANUAL
    elif navigation == "Prediksi Manual":
        st.title("Input Data Manual")
        st.write("Masukkan data pelanggan tunggal untuk melakukan pengecekan instan.")
        
        with st.form("manual_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Usia", 18, 100, 30)
                gender = st.selectbox("Gender", ["Male", "Female"])
                total_spent = st.number_input("Total Spent", 0.0, 10000.0, 500.0)
            with c2:
                satisfaction = st.slider("Satisfaction Score", 1.0, 5.0, 3.0)
                sessions = st.number_input("Avg Session Time", 0.0, 100.0, 10.0)
                tickets = st.number_input("Support Tickets", 0, 20, 1)
            with c3:
                pages = st.number_input("Pages per Session", 0.0, 50.0, 5.0)
                lvt = st.number_input("Lifetime Value", 0.0, 20000.0, 1000.0)
                email = st.slider("Email Open Rate", 0.0, 1.0, 0.5)
            
            submit = st.form_submit_button("Analisis Sekarang")
            
        if submit:
            input_dict = {
                'age': [age], 'gender': [gender], 'total_spent': [total_spent],
                'satisfaction_score': [satisfaction], 'avg_session_time': [sessions],
                'support_tickets': [tickets], 'pages_per_session': [pages],
                'lifetime_value': [lvt], 'email_open_rate': [email]
            }
            df_man = pd.DataFrame(input_dict)
            df_man_enc = pd.get_dummies(df_man, drop_first=True).reindex(columns=columns_list, fill_value=0)
            df_man_sc = scaler.transform(df_man_enc)
            
            res = model.predict(df_man_sc)[0]
            prob = model.predict_proba(df_man_sc)[0][1]
            
            if res == 1:
                st.error(f"Hasil Analisis: Pelanggan ini berisiko CHURN sebesar {prob*100:.1f}%")
            else:
                st.success(f"Hasil Analisis: Pelanggan ini diprediksi RETAIN (Setia) dengan probabilitas {(1-prob)*100:.1f}%")

    # HALAMAN 4: INFORMASI MODEL
    elif navigation == "Informasi Model":
        st.title("Keterangan Teknis")
        st.write("Aplikasi ini menggunakan model Random Forest yang telah dioptimasi dengan Hyperparameter Tuning.")
        st.markdown("""
        - **Model:** Random Forest Classifier
        - **Scaling:** StandardScaler
        - **Fitur Utama:** Satisfaction Score, Total Spent, Support Tickets.
        - **Versi:** 2.0 (UAS Final Edition)
        """)
