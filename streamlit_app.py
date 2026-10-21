import streamlit as st
import prediction
import eda

st.set_page_config(
    page_title='Bank Customer Churn Prediction',
    layout='wide',
    initial_sidebar_state='expanded'
)

page = st.sidebar.selectbox('Pilih Halaman', ['EDA', 'Prediction'])

if page == 'EDA':
    eda.run()
else:
    prediction.run()