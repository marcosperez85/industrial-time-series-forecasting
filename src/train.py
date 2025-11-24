import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

def train():

    df = pd.read_csv("../data/raw/gas_consumption.csv", parse_dates=["date"])

    df["hour"] = df["date"].dt.hour

    X = df[["temp", "humidity", "wind", "hour"]]
    y = df["consumption_m3"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, "../models/model.pkl")

    print("Modelo entrenado y guardado en models/model.pkl")

if __name__ == "__main__":
    train()
