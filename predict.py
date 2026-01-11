import cv2
import numpy as np
from tensorflow import keras
from huggingface_hub import hf_hub_download

# download model from HuggingFace
model_path = hf_hub_download(
    repo_id="samikshachougule-hub/emotion-recognition-model",
    filename="model_file_30epochs.h5"
)

model = keras.models.load_model(model_path, compile=False)

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

def predict_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)

    if len(faces):
        x,y,w,h = faces[0]
        crop = gray[y:y+h, x:x+w]
        crop = cv2.resize(crop,(48,48))
        crop = crop / 255.0
        crop = np.reshape(crop,(1,48,48,1))
        preds = model.predict(crop, verbose=0)
        label = labels_dict[np.argmax(preds)]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    return frame
