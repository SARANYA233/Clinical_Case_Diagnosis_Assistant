import joblib

SYMPTOM_COLUMNS = joblib.load("./models/symptom_columns.pkl")
print(SYMPTOM_COLUMNS)