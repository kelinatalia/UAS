"""
app_streamlit_cloud.py  —  soal 2.b

Sama persis dengan app_streamlit.py (local), satu-satunya perbedaan:
import dari inferencing_cloud (bukan inferencing), supaya model
di-load dari S3 kalau belum ada di lokal.

Cara jalanin di EC2:
    export S3_BUCKET=nama-bucket-lu
    streamlit run app_streamlit_cloud.py --server.port 8501 --server.address 0.0.0.0
"""

import streamlit as st
import pandas as pd
from inferencing_cloud import InferenceService, TEST_CASES, ALL_FEATURES
import os

st.set_page_config(page_title="Credit Score Predictor (Cloud)", layout="wide")


@st.cache_resource
def load_service():
    try:
        bucket = os.environ.get('S3_BUCKET', 'GANTI_NAMA_BUCKET_LU')
        return InferenceService(bucket=bucket)
    except Exception as e:
        return e


service = load_service()

st.sidebar.title("Informasi Aplikasi")
st.sidebar.info(
    "Aplikasi prediksi Credit Score nasabah (Good / Standard / Poor). "
    "Model di-load dari AWS S3."
)

if isinstance(service, Exception):
    st.sidebar.error(str(service))
    st.error(
        f"Gagal load model dari S3: {service}\n\n"
        "Pastikan:\n"
        "1. Environment variable `S3_BUCKET` sudah di-set\n"
        "2. EC2 punya IAM role dengan akses S3\n"
        "3. pipeline_cloud.py sudah dijalankan dan model sudah ke-upload ke S3"
    )
    st.stop()

st.title("Credit Score Prediction Dashboard")
st.caption("Model di-load dari AWS S3. Isi data nasabah atau pakai test case di sidebar.")

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Test Case")
st.sidebar.caption("Representative profile per kelas dari data training asli.")

preset_choice = st.sidebar.selectbox("Pilih test case", ["-- manual input --", "Good", "Standard", "Poor"])
if st.sidebar.button("Terapkan test case ini"):
    if preset_choice in TEST_CASES:
        for k, v in TEST_CASES[preset_choice].items():
            st.session_state[k] = v
        st.rerun()

DEFAULTS = TEST_CASES['Standard']


def default_of(field):
    return st.session_state.get(field, DEFAULTS[field])


tab1, tab2 = st.tabs(["Demografi & Pemasukan", "Rekening, Kartu & Pinjaman"])

with tab1:
    col1, col2 = st.columns(2)
    MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August"]
    JOBS   = ['Accountant', 'Architect', 'Developer', 'Doctor', 'Engineer', 'Entrepreneur',
              'Journalist', 'Lawyer', 'Manager', 'Mechanic', 'Media_Manager', 'Musician',
              'Scientist', 'Teacher', 'Writer']
    with col1:
        st.selectbox("Bulan", MONTHS, key="Month",
                     index=MONTHS.index(default_of("Month")))
        st.number_input("Usia", min_value=14, max_value=100, key="Age",
                         value=int(default_of("Age")))
        st.selectbox("Pekerjaan", JOBS, key="Occupation",
                     index=JOBS.index(default_of("Occupation")))
    with col2:
        st.number_input("Pendapatan Tahunan ($)", min_value=0.0, key="Annual_Income",
                         value=float(default_of("Annual_Income")), step=1000.0)
        st.number_input("Gaji Bersih Bulanan ($)", min_value=0.0, key="Monthly_Inhand_Salary",
                         value=float(default_of("Monthly_Inhand_Salary")), step=100.0)
        st.number_input("Saldo Akhir Bulanan ($)", min_value=0.0, key="Monthly_Balance",
                         value=float(default_of("Monthly_Balance")), step=50.0)

with tab2:
    c1, c2, c3 = st.columns(3)
    MIXES = ["Good", "Standard", "Bad"]
    BEHAVIOURS = [
        "High_spent_Large_value_payments", "High_spent_Medium_value_payments",
        "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
        "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments"
    ]
    with c1:
        st.number_input("Jumlah Rekening Bank", 0, 20, key="Num_Bank_Accounts",
                         value=int(default_of("Num_Bank_Accounts")))
        st.number_input("Jumlah Kartu Kredit", 0, 20, key="Num_Credit_Card",
                         value=int(default_of("Num_Credit_Card")))
        st.number_input("Suku Bunga Rata-rata (%)", 0, 50, key="Interest_Rate",
                         value=int(default_of("Interest_Rate")))
        st.number_input("Total Jumlah Pinjaman", 0, 20, key="Num_of_Loan",
                         value=int(default_of("Num_of_Loan")))
        st.number_input("Jumlah Jenis Pinjaman Berbeda", 0, 10, key="Loan_Type_Count",
                         value=int(default_of("Loan_Type_Count")))
    with c2:
        st.number_input("Rata-rata Telat Bayar (Hari)", -10, 120, key="Delay_from_due_date",
                         value=int(default_of("Delay_from_due_date")))
        st.number_input("Banyaknya Pembayaran Telat", 0, 50, key="Num_of_Delayed_Payment",
                         value=int(default_of("Num_of_Delayed_Payment")))
        st.number_input("Perubahan Limit Kredit (%)", -20.0, 50.0, key="Changed_Credit_Limit",
                         value=float(default_of("Changed_Credit_Limit")))
        st.number_input("Banyaknya Inquiry Kredit", 0, 30, key="Num_Credit_Inquiries",
                         value=int(default_of("Num_Credit_Inquiries")))
        st.selectbox("Mix Kredit", MIXES, key="Credit_Mix",
                     index=MIXES.index(default_of("Credit_Mix")))
    with c3:
        st.number_input("Hutang Belum Dibayar ($)", 0.0, key="Outstanding_Debt",
                         value=float(default_of("Outstanding_Debt")))
        st.slider("Rasio Penggunaan Kredit (%)", 10.0, 60.0, key="Credit_Utilization_Ratio",
                  value=float(default_of("Credit_Utilization_Ratio")))
        st.number_input("Lama Riwayat Kredit (Bulan)", 0, key="Credit_History_Months",
                         value=int(default_of("Credit_History_Months")))
        st.selectbox("Bayar Nominal Minimum?", ["Yes", "No"], key="Payment_of_Min_Amount",
                     index=["Yes", "No"].index(default_of("Payment_of_Min_Amount")))
        st.number_input("Total EMI Bulanan ($)", 0.0, key="Total_EMI_per_month",
                         value=float(default_of("Total_EMI_per_month")))
        st.number_input("Investasi Bulanan ($)", 0.0, key="Amount_invested_monthly",
                         value=float(default_of("Amount_invested_monthly")))
        st.selectbox("Perilaku Pembayaran", BEHAVIOURS, key="Payment_Behaviour",
                     index=BEHAVIOURS.index(default_of("Payment_Behaviour")))

st.markdown("---")
if st.button("Prediksi Credit Score", type="primary", use_container_width=True):
    input_dict = {f: st.session_state[f] for f in ALL_FEATURES}
    with st.spinner("Menganalisis..."):
        try:
            result     = service.predict_one(input_dict)
            prediction = result['prediction']

            if prediction == "Good":
                st.success(f"### Hasil Prediksi: {prediction} Credit Score")
            elif prediction == "Standard":
                st.info(f"### Hasil Prediksi: {prediction} Credit Score")
            else:
                st.error(f"### Hasil Prediksi: {prediction} Credit Score")

            if 'probabilities' in result:
                st.subheader("Probabilitas per Kelas")
                proba_df = pd.DataFrame({
                    'Kelas': list(result['probabilities'].keys()),
                    'Probabilitas': list(result['probabilities'].values())
                }).sort_values('Probabilitas', ascending=False)
                st.bar_chart(proba_df.set_index('Kelas'))
                st.dataframe(proba_df, hide_index=True, use_container_width=True)

        except Exception as e:
            st.error(f"Error prediksi: {e}")
