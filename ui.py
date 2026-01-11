import streamlit as st
import cv2
from predict import predict_frame

st.title("😃 Real-time Emotion Recognition")

run = st.checkbox("Start Webcam")
frame_holder = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera access failed")
        break

    frame = predict_frame(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_holder.image(frame)

cap.release()
