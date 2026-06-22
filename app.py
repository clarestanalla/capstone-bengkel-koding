import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction")

st.write("""
Aplikasi ini digunakan untuk memprediksi apakah pelanggan akan **Churn**
atau **Tidak Churn** menggunakan model Logistic Regression.
""")

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

uploaded_file = st.file_uploader(
    "Upload Dataset CSV",
    type="csv"
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset")

    st.dataframe(df.head())

    try:

        if "churn" in df.columns:
            df = df.drop("churn", axis=1)

        df = pd.get_dummies(
            df,
            drop_first=True
        )

        df = df.reindex(
            columns=columns,
            fill_value=0
        )

        X_scaled = scaler.transform(df)

        prediction = model.predict(X_scaled)

        df["Prediction"] = prediction

        df["Prediction"] = df["Prediction"].map({
            0: "Not Churn",
            1: "Churn"
        })

        st.success("Prediction Success")

        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Prediction",
            csv,
            "prediction.csv",
            "text/csv"
        )

    except Exception as e:

        st.error(e)
