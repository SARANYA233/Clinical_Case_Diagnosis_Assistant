from dotenv import load_dotenv
from src.utils.symptom_mapper import URL, map_symptoms_groq
import joblib
import numpy as np
from src.exception import CustomException
import sys
import os
import gdown
import pickle
import requests

load_dotenv()

RF_MODEL = "1O33vbPu10sDn3tXHviz9XaS0kAnZK4tc"
SYMPTOM_COLUMNS = "1E_Vu_Dw5lBLhSqsfbdP2xqaVH88A68nd"

os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/symptom_model.pkl"
COLUMNS_PATH = "models/symptom_columns.pkl"

def download_model():
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists(MODEL_PATH):
        URL = "https://drive.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params={'id': RF_MODEL}, stream=True)
        # gdown.download(f"https://drive.google.com/uc?id={RF_MODEL}", MODEL_PATH, quiet=False)
    if not os.path.exists(COLUMNS_PATH):
        URL = "https://drive.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params={'id': SYMPTOM_COLUMNS}, stream=True)
        # gdown.download(f"https://drive.google.com/uc?id={SYMPTOM_COLUMNS}", COLUMNS_PATH, quiet=False)

def load_model():
    download_model()
    with open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)
    with open(COLUMNS_PATH, "rb") as f:
        columns = joblib.load(f)
    return model, columns

def predict_disease(user_input, top_k = 3):
    try:

        model, columns = load_model()
        symptoms = map_symptoms_groq(user_input)

        input_vector = [1 if symptom in symptoms else 0 for symptom in columns]
        input_vector = np.array(input_vector).reshape(1, -1)

        probabilities = model.predict_proba(input_vector)[0]
        indices = probabilities.argsort()[::-1][:top_k]
        diseases = [(model.classes_[i], round(probabilities[i] * 100, 2)) for i in indices]

        return diseases, symptoms
    
    except Exception as e:
        raise CustomException(e, sys)
    