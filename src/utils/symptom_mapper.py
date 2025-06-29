from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import joblib
from src.exception import CustomException
import sys
import pickle
import streamlit as st

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = ChatGroq(model="Gemma2-9b-It", groq_api_key=GROQ_API_KEY)
with open("./models/symptom_columns.pkl", "rb") as f:
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
        st.write("Creating chain with prompt and model...")
        st.write(MODEL)

        chain = prompt | MODEL | parser
        response = chain.invoke({
            "user_input": user_input,
            "SYMPTOM_COLUMNS": SYMPTOM_COLUMNS
        })
        st.write("Response from model:", response)
        extracted_symptoms = response.split(',')
        result = [symptom.replace('\n', '').strip() for symptom in extracted_symptoms if symptom.replace('\n', '').strip() in SYMPTOM_COLUMNS]
        return result
    except Exception as e:
        st.write("An error occurred while mapping symptoms.")
        st.write(f"Error: {str(e)}")
        st.error(f"Error during prediction: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        raise CustomException(e, sys)
    

def map_symptoms_local():
    pass

if __name__ == "__main__":
    test_input = "I feel a tight pressure in my chest, trouble sleeping, and dizziness when I stand up."

    result = map_symptoms_groq(test_input)
    print("Matched Symptoms:", result)