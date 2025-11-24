import joblib
import pandas as pd

def predict(temp, humidity, wind, hour):
    model = joblib.load("../models/model.pkl")
    X = pd.DataFrame([[temp, humidity, wind, hour]], 
                     columns=["temp", "humidity", "wind", "hour"])
    return model.predict(X)[0]

if __name__ == "__main__":
    result = predict(28, 60, 12, 14)
    print("Predicción:", result)
