import io
import numpy as np
from fastapi import FastAPI, UploadFile
from tensorflow.keras.models import load_model
from PIL import Image

app = FastAPI()

@app.get("/")
def greet():
    return {"message": "bonjour"}

def load():
    model_path = "best_model.h5"
    model = load_model(model_path, compile=False)
    return model

# Chargement du modèle 
model = load()

def preprocess(img):
    img = img.resize((224, 224))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    return img


@app.post("/predict")
async def predict(file: UploadFile):
    img_data = await file.read()

    # ouvrir l'image
    img = Image.open(io.BytesIO(img_data))

    # preprocessing
    img_processed = preprocess(img)

    # prédiction
    predictions = model.predict(img_processed)
    rec = predictions[0][0].tolist()

    return {"Prédictions": rec}