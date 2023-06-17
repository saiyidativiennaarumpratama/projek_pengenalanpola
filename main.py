import streamlit as st
import joblib
from PIL import Image

image = Image.open('assets/logo-jne.png')
st.image(image, use_column_width=True)
st.markdown("<h1 style='text-align: center;'>ANALISIS SENTIMEN OPINI MASYARAKAT TERHADAP JASA EKSPEDISI JNE MENGGUNAKAN ALGORITMA NAÏVE BAYES</h1>", unsafe_allow_html=True)
st.write("##### Saiyidati Vienna Arum Pratama 200411100018")
st.write("##### Proyek Akhir Pengenalan Pola")

# load model
model = joblib.load('model/bestmodelNBMultinomial.pkl')
# load vectorizer
vectorizer = joblib.load('model/proses_tf-idf.pkl')

# inputan
a = st.text_input('Masukkan ulasan')

if st.button('Predict'):
    # transform input menggunakan vectorizer
    transform = vectorizer.transform([a])

    # prediksi menggunakan model
    pred = model.predict(transform)

    # tampilkan hasil prediksi
    st.write("Sentiment: ", pred[0])

