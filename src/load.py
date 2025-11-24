import pandas as pd

def load_data(path="../data/raw/gas_consumption.csv"):
    return pd.read_csv(path, parse_dates=["date"])
