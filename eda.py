import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run():
    '''
    Fungsi untuk menjalankan dashboard EDA
    '''
    st.title('Exploratory Data Analysis - Bank Customer Churn')

    st.subheader('Dashboard Analisis Karakteristik & Perilaku Nasabah')

    data_path = 'Customer-Churn-Records.csv'
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.replace(' ', '', regex=False)
    
    st.write('### Sampel Data Nasabah')
    st.dataframe(df.head(10))

    # Visualisasi 1: Distribusi Target (Exited)
    st.write('### 1. Distribusi Target Exited (Churn vs Loyal)')
    fig1 = plt.figure(figsize=(10, 4.5))
    sns.countplot(data=df, x='Exited', palette=['#2b5c8f', '#d95f02'])
    plt.xticks([0, 1], ['0 - Loyal (79.62%)', '1 - Churn (20.38%)'])
    plt.title('Distribusi Nasabah Churn')
    st.pyplot(fig1)

    # Visualisasi 2: Hubungan Jumlah Produk terhadap Churn Rate
    st.write('### 2. Tingkat Churn (%) Berdasarkan Jumlah Produk')
    fig2 = plt.figure(figsize=(10, 4.5))
    prod_churn = df.groupby('NumOfProducts')['Exited'].mean() * 100
    sns.barplot(x=prod_churn.index, y=prod_churn.values, palette='magma')
    plt.ylabel('Churn Rate (%)')
    plt.xlabel('Jumlah Produk (NumOfProducts)')
    plt.title('Tingkat Churn per Jumlah Produk')
    st.pyplot(fig2)

    # Visualisasi 3: Usia (Age) Berdasarkan Status Churn
    st.write('### 3. Distribusi Usia (Age) Nasabah Churn vs Loyal')
    fig3 = plt.figure(figsize=(10, 4.5))
    sns.histplot(data=df, x='Age', hue='Exited', kde=True, bins=30, palette=['#2b5c8f', '#d95f02'])
    plt.title('Distribusi Usia Berdasarkan Status Churn')
    st.pyplot(fig3)

    # Visualisasi 4: Churn Rate per Negara (Geography)
    st.write('### 4. Tingkat Churn (%) Berdasarkan Negara (Geography)')
    fig4 = plt.figure(figsize=(10, 4.5))
    geo_churn = df.groupby('Geography')['Exited'].mean() * 100
    sns.barplot(x=geo_churn.index, y=geo_churn.values, palette=['#377eb8', '#e41a1c', '#4daf4a'])
    plt.ylabel('Churn Rate (%)')
    plt.title('Tingkat Churn Berdasarkan Negara')
    st.pyplot(fig4)

    # Plot Interaktif (Sesuai Pilihan Kolom User)
    st.write('### 5. Eksplorasi Fitur Numerik (Interaktif)')
    opsi = st.selectbox('Pilih Kolom Numerik:', ('CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary', 'PointEarned'))
    fig5 = plt.figure(figsize=(10, 4.5))
    sns.histplot(df[opsi], bins=30, kde=True, color='#2b5c8f')
    plt.title(f'Distribusi Kolom {opsi}')
    st.pyplot(fig5)

if __name__ == '__main__':
    run()