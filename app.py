 # app.py

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import pickle
import pandas as pd

app = FastAPI()

# Load trained pipeline model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


@app.get('/')
def get():
    return RedirectResponse(url='/docs')

@app.post("/predict")
def predict(data: dict):

    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([data])

    # Prediction
    prediction = model.predict(input_df)

    return {"Predicted Life Expectancy": float(prediction[0])}
