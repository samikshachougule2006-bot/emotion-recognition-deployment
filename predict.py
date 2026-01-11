import onnxruntime as ort
import numpy as np
import cv2
from huggingface_hub import hf_hub_download

# download ONNX model from HF Hub
model_path = hf_hub_download(
    repo_id="samikshachougule-hub/emotion-recognition-model",
    filename="model.onnx"
)

# load ONNX session
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

def predict_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)

    if len(faces):
        x,y,w,h = faces[0]
        crop = gray[y:y+h, x:x+w]
        crop = cv2.resize(crop,(48,48))
        crop = crop / 255.0
        crop = np.reshape(crop,(1,48,48,1)).astype(np.float32)

        preds = session.run([output_name], {input_name: crop})[0]
        label = labels_dict[np.argmax(preds)]
        return label

    return "No Face"
