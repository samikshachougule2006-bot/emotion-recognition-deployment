import streamlit as st
import cv2
import numpy as np
from predict import predict_frame

st.title("😃 Facial Emotion Recognition")

mode = st.radio("Select Mode", ["Upload Image (Cloud)", "Webcam (Local)"])

# ---------------------
# MODE 1: UPLOAD (Cloud)
# ---------------------
if mode == "Upload Image (Cloud)":
    uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        result = predict_frame(img)

        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, caption="Prediction Output")

# ---------------------
# MODE 2: WEBCAM (Local)
# ---------------------
else:
    run = st.checkbox("Start Webcam")
    frame_holder = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not accessible")
            break

        frame = predict_frame(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_holder.image(frame_rgb)

    cap.release()
