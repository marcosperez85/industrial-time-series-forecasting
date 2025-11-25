import pandas as pd

def load_data(path="../data/raw/industrial_timeseries.csv"):
    """
    Carga el dataset de series temporales industriales.
    
    Args:
        path (str): Ruta al archivo CSV
        
    Returns:
        pd.DataFrame: Dataset con timestamp parseado como datetime
    """
    return pd.read_csv(path, parse_dates=["timestamp"])

def load_processed_data(path="../data/processed/industrial_timeseries_featured.csv"):
    """
    Carga el dataset procesado con features de ingeniería.
    
    Args:
        path (str): Ruta al archivo CSV procesado
        
    Returns:
        pd.DataFrame: Dataset con features engineered
    """
    return pd.read_csv(path, parse_dates=["timestamp"])