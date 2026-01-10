import streamlit as st
import cv2
import requests
import numpy as np

BACKEND_URL = "http://127.0.0.1:8000/predict"

st.title("😃 Real-time Emotion Recognition")

run = st.checkbox('Start Webcam')

frame_container = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()

    if not ret:
        st.error("Failed to access webcam")
        break

    _, buf = cv2.imencode(".jpg", frame)

    resp = requests.post(
        BACKEND_URL,
        files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")}
    )

    arr = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    frame_container.image(img)

cap.release()