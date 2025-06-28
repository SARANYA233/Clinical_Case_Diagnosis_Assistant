from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import joblib
from src.exception import CustomException
import sys
import pickle
import gdown

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = ChatGroq(model="Gemma2-9b-It", groq_api_key=GROQ_API_KEY)
# SYMPTOM_COLUMNS = joblib.load("D:/PROJECTS/Clinical_Case_Diagnosis_Assistant/models/symptom_columns.pkl")
COLUMNS_PATH = "models/symptom_columns.pkl"
SYMPTOM_COLUMNS = "1E_Vu_Dw5lBLhSqsfbdP2xqaVH88A68nd"
if not os.path.exists(COLUMNS_PATH):
    gdown.download(f"https://drive.google.com/uc?id={SYMPTOM_COLUMNS}", COLUMNS_PATH, quiet=False)

with open(COLUMNS_PATH, "rb") as f:
    SYMPTOM_COLUMNS = pickle.load(f)

def map_symptoms_groq(user_input):
    try:
        parser = StrOutputParser()
        System_prompt = """
            You are a medical assistant. Based on the user's symptom description, identify relevant symptoms that exactly match the list below.

            Only return a string of exact symptom matches from this list seperated by commas:
            {SYMPTOM_COLUMNS}

        """
        User_prompt = """ Patient description: {user_input} """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", System_prompt),
                ("user", User_prompt)
            ]
        )

        chain = prompt | MODEL | parser
        response = chain.invoke({
            "user_input": user_input,
            "SYMPTOM_COLUMNS": SYMPTOM_COLUMNS
        })

        extracted_symptoms = response.split(',')
        result = [symptom.replace('\n', '').strip() for symptom in extracted_symptoms if symptom.replace('\n', '').strip() in SYMPTOM_COLUMNS]
        return result
    except Exception as e:
        raise CustomException(e, sys)

def map_symptoms_local():
    pass

if __name__ == "__main__":
    test_input = "I feel a tight pressure in my chest, trouble sleeping, and dizziness when I stand up."

    result = map_symptoms_groq(test_input)
    print("Matched Symptoms:", result)