from joblib import load
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris

# Chargement des données
iris= load_iris()

# Chargement du modèle
loaded_model = load('logreg.joblib')

# Créer une instance de FastAPI
app = FastAPI()

# Définir un objet pour faire des requêtes
class request_body(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Définition du point de terminaison
@app.post("/predict") #local : http://127.0.0.1:8000/predict

# Fonction de prédiction
def predict(data : request_body):
    # Nouvelles données pour la prédiction
    new_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    # Prédiction
    class_idx = loaded_model.predict(new_data)[0]

    # On retourne le nom de l'espèce iris
    return {'class' : iris.target_names[class_idx]}