# 😃 Real-Time Facial Emotion Recognition

This project performs **real-time emotion detection** from webcam video using a **CNN model trained on FER-2013**.  
It identifies **7 emotions**:

> Angry · Disgust · Fear · Happy · Neutral · Sad · Surprise

---

## 🚀 Features

✔ Real-time webcam processing  
✔ CNN-based model inference  
✔ Bounding box & emotion labels  
✔ Lightweight inference, no GPU required  
✔ Clean Streamlit UI  
✔ Model hosted via HuggingFace Hub  
✔ Deployable to Streamlit Cloud

---

## 🧠 Model Details

- Dataset: **FER-2013**
- Framework: **TensorFlow + Keras**
- Input size: `48x48 grayscale`
- Output classes: `7`

Model file stored on HuggingFace:

🔗 **Model Link:**  
https://huggingface.co/samikshachougule-hub/emotion-recognition-model/blob/main/model_file_30epochs.h5

---

## 🖥️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Frontend UI | Streamlit |
| Inference | TensorFlow / Keras |
| Image Processing | OpenCV |
| Deployment | Streamlit Cloud |
| Model Hosting | HuggingFace Hub |

---

## 📂 Project Structure

emotion-recognition/
│
├── ui.py # Streamlit app
├── predict.py # Model inference logic
├── haarcascade_frontalface_default.xml
├── requirements.txt
└── README.md
👩‍💻 Author

Samiksha Chougule

🔗 GitHub: 
🔗 HuggingFace: https://huggingface.co/samikshachougule-hub

💼 Use Case Applications

✔ E-learning engagement tracking
✔ Mental health analysis
✔ Customer satisfaction kiosks
✔ Market research & UX testing
✔ Human-computer interaction
