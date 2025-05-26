import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("student_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
X_train_columns = joblib.load("X_train_columns.pkl")

st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", layout="centered")
st.title("🎓 Prediksi Status Mahasiswa: Graduate atau Dropout")

st.subheader("📋 Data Pendaftaran")
course = st.selectbox("Program Studi (Course)", [33, 171, 9070, 9254])
attendance = st.selectbox("Jadwal Kuliah", [1, 0], format_func=lambda x: "Siang" if x == 1 else "Malam")
previous_qualification = st.selectbox("Kualifikasi Sebelumnya", [1, 2, 4, 5, 6, 8, 9, 10, 12, 14, 15, 19])
prev_qualification_grade = st.number_input("Nilai Kualifikasi Sebelumnya", min_value=0.0, max_value=200.0, step=1.0)
admission_grade = st.number_input("Nilai Masuk (Admission Grade)", min_value=0.0, max_value=200.0, step=1.0)
debtor = st.selectbox("Tunggakan Pembayaran?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
tuition_up_to_date = st.selectbox("Uang Kuliah Terbayar?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
scholarship_holder = st.selectbox("Penerima Beasiswa?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
age_enroll = st.slider("Usia Saat Masuk Kuliah", 17, 60, 18)
international = st.selectbox("Mahasiswa Internasional?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

st.subheader("📚 Data Akademik Semester 1")
sem1_enrolled = st.slider("Mata Kuliah Diambil Semester 1", 0, 20, 6)
sem1_eval = st.slider("Dievaluasi Semester 1", 0, 20, 6)
sem1_approved = st.slider("Lulus Semester 1", 0, 20, 6)
sem1_grade = st.slider("Rata-rata Nilai Semester 1", 0.0, 20.0, 14.0)
sem1_without_eval = st.slider("Tanpa Evaluasi Semester 1", 0, 10, 0)

st.subheader("📚 Data Akademik Semester 2")
sem2_enrolled = st.slider("Mata Kuliah Diambil Semester 2", 0, 20, 6)
sem2_eval = st.slider("Dievaluasi Semester 2", 0, 20, 6)
sem2_approved = st.slider("Lulus Semester 2", 0, 20, 6)
sem2_grade = st.slider("Rata-rata Nilai Semester 2", 0.0, 20.0, 14.0)
sem2_without_eval = st.slider("Tanpa Evaluasi Semester 2", 0, 10, 0)

st.subheader("📉 Data Ekonomi Global")
unemployment_rate = st.number_input("Tingkat Pengangguran (%)", value=10.0)
inflation_rate = st.number_input("Tingkat Inflasi (%)", value=1.0)
gdp = st.number_input("GDP (%)", value=1.5)

if st.button("🔮 Prediksi"):
    input_df = pd.DataFrame([[
        course,
        attendance,
        previous_qualification,
        prev_qualification_grade,
        admission_grade,
        debtor,
        tuition_up_to_date,
        scholarship_holder,
        age_enroll,
        international,
        sem1_enrolled,
        sem1_eval,
        sem1_approved,
        sem1_grade,
        sem1_without_eval,
        sem2_enrolled,
        sem2_eval,
        sem2_approved,
        sem2_grade,
        sem2_without_eval,
        unemployment_rate,
        inflation_rate,
        gdp
    ]], columns=X_train_columns)

    # Encode kolom kategori pakai label_encoders
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    input_df = input_df[X_train_columns]

    # Prediksi
    prediction = model.predict(input_df)[0]
    label = "🎓 Graduate" if prediction == 1 else "❌ Dropout"
    st.success(f"**Status Mahasiswa yang Diprediksi: {label}**")
