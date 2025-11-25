from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Optional
import os

app = FastAPI(
    title="Industrial Time Series Forecasting API",
    description="API para predicción de valores en series temporales industriales",
    version="1.0.0"
)

# Cargar modelo y configuración
try:
    model = joblib.load("../models/best_model.pkl")
    model_info = joblib.load("../models/model_info.pkl")
    print(f"Modelo cargado: {model_info['model_type']}")
except FileNotFoundError:
    print("⚠️  Modelo no encontrado. Ejecuta primero el notebook 04_model.ipynb")
    model = None
    model_info = None

class PredictRequest(BaseModel):
    temperature: float
    demand_factor: float  # 0-1
    operational_efficiency: float  # 0-1  
    energy_price: float
    hour: int  # 0-23
    day_of_week: int  # 0-6 (0=Monday)
    month: int  # 1-12
    is_weekend: int  # 0 or 1
    lag_1h: Optional[float] = None
    lag_24h: Optional[float] = None
    rolling_mean_24h: Optional[float] = None
    rolling_std_24h: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "temperature": 22.5,
                "demand_factor": 0.75,
                "operational_efficiency": 0.85,
                "energy_price": 85.0,
                "hour": 14,
                "day_of_week": 2,
                "month": 6,
                "is_weekend": 0,
                "lag_1h": 1150.0,
                "lag_24h": 1180.0,
                "rolling_mean_24h": 1165.0,
                "rolling_std_24h": 25.0
            }
        }

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Realiza una predicción basada en las features proporcionadas.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # Crear features adicionales
        temp_squared = req.temperature ** 2
        demand_efficiency_interaction = req.demand_factor * req.operational_efficiency
        
        # Valores por defecto para features opcionales
        lag_1h = req.lag_1h if req.lag_1h is not None else 1100.0
        lag_24h = req.lag_24h if req.lag_24h is not None else 1100.0
        rolling_mean_24h = req.rolling_mean_24h if req.rolling_mean_24h is not None else 1100.0
        rolling_std_24h = req.rolling_std_24h if req.rolling_std_24h is not None else 20.0
        
        # Preparar datos en el orden correcto según model_info
        feature_values = {
            'temperature': req.temperature,
            'demand_factor': req.demand_factor,
            'operational_efficiency': req.operational_efficiency,
            'energy_price': req.energy_price,
            'hour': req.hour,
            'day_of_week': req.day_of_week,  
            'month': req.month,
            'is_weekend': req.is_weekend,
            'lag_1h': lag_1h,
            'lag_24h': lag_24h,
            'rolling_mean_24h': rolling_mean_24h,
            'rolling_std_24h': rolling_std_24h,
            'temp_squared': temp_squared,
            'demand_efficiency_interaction': demand_efficiency_interaction
        }
        
        # Crear array de features en el orden correcto
        feature_array = np.array([[feature_values[col] for col in model_info['features']]])
        
        # Realizar predicción
        prediction = model.predict(feature_array)[0]
        
        return {
            "prediction": round(prediction, 2),
            "model_type": model_info['model_type'],
            "model_mae": round(model_info['test_mae'], 2),
            "features_used": model_info['features']
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {str(e)}")

@app.get("/")
def root():
    return {
        "message": "API para predicción de series temporales industriales",
        "model_loaded": model is not None,
        "endpoints": ["/predict", "/health", "/model-info"]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/model-info")
def model_info_endpoint():
    if model_info is None:
        raise HTTPException(status_code=503, detail="Información del modelo no disponible")
    
    return {
        "model_type": model_info['model_type'],
        "test_mae": model_info['test_mae'],
        "features_count": len(model_info['features']),
        "required_features": model_info['features']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)