import pandas as pd
from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel
from deep_translator import GoogleTranslator
from pretraitement import cleaned_text
from fastapi import HTTPException


df =  pd.read_csv('df_c.csv')

# Charger le modèle et le vectoriseur
model = load('new_model.joblib')
vectorizer = load('vectoriz.joblib')

# Création d'une nouvelle instance fastAPI
app = FastAPI()

@app.get("/") #
def Accueil ():
    return {"message": "Bienvenu sur l'appli de prediction des sentiments"} 

# Définir un objet (une classe) pour réaliser des requêtes
class request_body(BaseModel):
    text_tweet : str
    

# Definition du chemin du point de terminaison (API)

@app.post("/predict")  # uvicorn main:app --reload
def predict(data: request_body):
    try:
        translated_text = GoogleTranslator(source="fr", target="en").translate(data.text_tweet)

        clean_text = cleaned_text(translated_text)

        text_vector = vectorizer.transform([clean_text])

        # Prédiction 
        prediction = model.predict(text_vector)

        # Retourner si le tweet est positif ou négatif
        return {"sentiment": "Positive" if prediction == 1 else "Negative"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





