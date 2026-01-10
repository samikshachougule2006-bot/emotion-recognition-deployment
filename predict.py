import cv2
import numpy as np
from keras.models import load_model

model = load_model('model_file_30epochs.h5')
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict = {0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

def predict_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        sub = gray[y:y+h, x:x+w]
        sub = cv2.resize(sub, (48, 48))
        sub = sub / 255.0
        sub = np.reshape(sub, (1, 48, 48, 1))

        result = model.predict(sub, verbose=0)
        label = labels_dict[np.argmax(result)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    return frame