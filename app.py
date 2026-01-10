from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import numpy as np
import cv2
from predict import predict_frame

app = FastAPI()

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    img_bytes = await file.read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    output = predict_frame(frame)

    _, jpeg = cv2.imencode('.jpg', output)
    return Response(content=jpeg.tobytes(), media_type='image/jpeg')

@app.get("/")
def index():
    return {"status": "backend running"}