from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("models/model.pkl")

@app.get("/predict")
def predict(temp: float, humidity: float, wind: float, hour: int):
    X = pd.DataFrame([[temp, humidity, wind, hour]],
                     columns=["temp","humidity","wind","hour"])
    return {"prediction": model.predict(X)[0]}