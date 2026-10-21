# Import libraries
import pickle
import pandas as pd
import numpy as np
import streamlit as st

# Load Model
model_path = 'best_churn_model.pkl'
with open(model_path, 'rb') as file_1:
    best_churn_model = pickle.load(file_1)

def run():
    '''
    Fungsi untuk menjalankan form dan proses prediksi data inference
    '''
    st.title('Bank Customer Churn Prediction')
    st.subheader('Formulir Prediksi Churn Nasabah & Rekomendasi CRM')

    with st.form(key='churn_form'):
        name = st.text_input('Nama Belakang (Surname)', value='Pratama')
        customer_id = st.number_input('Customer ID', value=15600000)
        geography = st.selectbox('Negara Domisili (Geography)', ('France', 'Germany', 'Spain'))
        gender = st.selectbox('Jenis Kelamin (Gender)', ('Female', 'Male'))
        age = st.number_input('Usia (Age)', min_value=18, max_value=100, value=35)
        tenure = st.number_input('Lama Menjadi Nasabah (Tenure - Tahun)', min_value=0, max_value=10, value=3)
        credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
        balance = st.number_input('Saldo Rekening (Balance)', min_value=0.0, max_value=300000.0, value=60000.0)
        num_of_products = st.selectbox('Jumlah Produk (NumOfProducts)', (1, 2, 3, 4), index=0)
        has_cr_card = st.selectbox('Memiliki Kartu Kredit (HasCrCard)', (1, 0), format_func=lambda x: 'Ya' if x == 1 else 'Tidak')
        is_active_member = st.selectbox('Status Keaktifan (IsActiveMember)', (1, 0), format_func=lambda x: 'Aktif' if x == 1 else 'Tidak Aktif')
        estimated_salary = st.number_input('Estimasi Gaji (EstimatedSalary)', min_value=0.0, max_value=300000.0, value=50000.0)
        complain = st.selectbox('Komplain Aktif (Complain)', (0, 1), format_func=lambda x: 'Tidak Ada Komplain' if x == 0 else 'Ada Komplain')
        satisfaction_score = st.selectbox('Skor Kepuasan (SatisfactionScore)', (1, 2, 3, 4, 5), index=2)
        card_type = st.selectbox('Tipe Kartu (CardType)', ('SILVER', 'GOLD', 'PLATINUM', 'DIAMOND'))
        point_earned = st.number_input('Poin Reward (PointEarned)', min_value=0, max_value=1500, value=400)
        
        submitted = st.form_submit_button('Predict')

    data_inf = {
        'RowNumber': 1,
        'CustomerId': customer_id,
        'Surname': name,
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary,
        'Complain': complain,
        'SatisfactionScore': satisfaction_score,
        'CardType': card_type,
        'PointEarned': point_earned
    }

    data_inf = pd.DataFrame([data_inf])

    st.write('### Data Nasabah yang Dimasukkan:')
    st.dataframe(data_inf)

    if submitted:
        y_pred = best_churn_model.predict(data_inf)
        y_prob = best_churn_model.predict_proba(data_inf)[0, 1] * 100

        st.markdown('---')
        if y_pred[0] == 1 or complain == 1:
            st.error(f'**Hasil Prediksi: Nasabah Berpotensi CHURN** (Probabilitas: **{y_prob:.2f}%**)')
            st.write('**Rekomendasi Tindakan CRM:**')
            if complain == 1:
                st.write('- Prioritas penyelesaian komplain nasabah dalam waktu < 24 jam.')
                st.write('- Berikan voucher kompensasi layanan dan keringanan biaya administrasi.')
            else:
                st.write('- Lakukan intervensi preventif silent churn dengan penawaran bunga kompetitif.')
                st.write('- Tawarkan cashback transaksi harian dan fitur auto-budgeting di aplikasi mobile.')
        else:
            st.success(f'**Hasil Prediksi: Nasabah LOYAL (Tidak Churn)** (Probabilitas Churn: **{y_prob:.2f}%**)')
            st.write('**Rekomendasi Tindakan Marketing:**')
            if num_of_products == 1:
                st.write('- Target utama program *cross-selling* untuk menambah produk ke-2.')
                st.write('- Tawarkan pembukaan kartu kredit bebas iuran atau tabungan berjangka.')
            else:
                st.write('- Pertahankan loyalitas dengan program *upgrade tier* kartu atau poin reward.')

if __name__ == '__main__':
    run()