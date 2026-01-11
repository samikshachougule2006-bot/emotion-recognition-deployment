import streamlit as st
import cv2
import numpy as np
from predict import predict_image

st.title("😃 Emotion Recognition (Image Upload)")

uploaded = st.file_uploader("Upload a face image", type=['jpg','jpeg','png'])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    result = predict_image(img)
    st.image(img, channels="BGR")
    st.write("Prediction:", result)
