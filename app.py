# app.py

from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# Load trained pipeline model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


@app.get("/")
def home():
    return {"message": "Life Expectancy Prediction API is running"}


@app.post("/predict")
def predict(data: dict):

    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([data])

    # Prediction
    prediction = model.predict(input_df)

    return {"Predicted Life Expectancy": float(prediction[0])}
