import pandas as pd
import numpy as np
import joblib
from typing import Optional, Dict, Any

def predict_single(
    temperature: float,
    demand_factor: float,
    operational_efficiency: float,
    energy_price: float,
    hour: int,
    day_of_week: int = 0,
    month: int = 1,
    is_weekend: int = 0,
    lag_1h: Optional[float] = None,
    lag_24h: Optional[float] = None,
    rolling_mean_24h: Optional[float] = None,
    rolling_std_24h: Optional[float] = None,
    model_path: str = "../models/best_model.pkl",
    model_info_path: str = "../models/model_info.pkl"
) -> Dict[str, Any]:
    """
    Realiza una predicción individual usando el modelo entrenado.
    
    Args:
        temperature: Temperatura en °C
        demand_factor: Factor de demanda (0-1)
        operational_efficiency: Eficiencia operacional (0-1)
        energy_price: Precio de energía ($/MWh)
        hour: Hora del día (0-23)
        day_of_week: Día de la semana (0=Lunes, 6=Domingo)
        month: Mes (1-12)
        is_weekend: 1 si es fin de semana, 0 si no
        lag_1h: Valor de hace 1 hora (opcional)
        lag_24h: Valor de hace 24 horas (opcional)
        rolling_mean_24h: Media móvil 24h (opcional)
        rolling_std_24h: Desviación estándar móvil 24h (opcional)
        model_path: Ruta al modelo guardado
        model_info_path: Ruta a la información del modelo
        
    Returns:
        Dict con la predicción y metadatos
    """
    try:
        # Cargar modelo y configuración
        model = joblib.load(model_path)
        model_info = joblib.load(model_info_path)
        
        # Valores por defecto para features opcionales
        lag_1h = lag_1h if lag_1h is not None else 1100.0
        lag_24h = lag_24h if lag_24h is not None else 1100.0
        rolling_mean_24h = rolling_mean_24h if rolling_mean_24h is not None else 1100.0
        rolling_std_24h = rolling_std_24h if rolling_std_24h is not None else 20.0
        
        # Crear features adicionales
        temp_squared = temperature ** 2
        demand_efficiency_interaction = demand_factor * operational_efficiency
        
        # Preparar features en el orden correcto
        feature_values = {
            'temperature': temperature,
            'demand_factor': demand_factor,
            'operational_efficiency': operational_efficiency,
            'energy_price': energy_price,
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'is_weekend': is_weekend,
            'lag_1h': lag_1h,
            'lag_24h': lag_24h,
            'rolling_mean_24h': rolling_mean_24h,
            'rolling_std_24h': rolling_std_24h,
            'temp_squared': temp_squared,
            'demand_efficiency_interaction': demand_efficiency_interaction
        }
        
        # Crear array de features
        feature_array = np.array([[feature_values[col] for col in model_info['features']]])
        
        # Realizar predicción
        prediction = model.predict(feature_array)[0]
        
        return {
            'prediction': round(prediction, 2),
            'model_type': model_info['model_type'],
            'model_mae': model_info['test_mae'],
            'features_used': model_info['features'],
            'input_features': feature_values
        }
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Modelo no encontrado. Ejecuta primero el entrenamiento. Detalles: {e}")
    except Exception as e:
        raise Exception(f"Error en la predicción: {e}")

def predict_batch(
    df: pd.DataFrame,
    model_path: str = "../models/best_model.pkl",
    model_info_path: str = "../models/model_info.pkl"
) -> pd.DataFrame:
    """
    Realiza predicciones en lote para un DataFrame.
    
    Args:
        df: DataFrame con las features necesarias
        model_path: Ruta al modelo guardado
        model_info_path: Ruta a la información del modelo
        
    Returns:
        DataFrame original con columna 'prediction' añadida
    """
    try:
        # Cargar modelo y configuración
        model = joblib.load(model_path)
        model_info = joblib.load(model_info_path)
        
        # Verificar que todas las features necesarias están presentes
        missing_features = set(model_info['features']) - set(df.columns)
        if missing_features:
            raise ValueError(f"Features faltantes en el DataFrame: {missing_features}")
        
        # Seleccionar features en el orden correcto
        X = df[model_info['features']]
        
        # Realizar predicciones
        predictions = model.predict(X)
        
        # Añadir predicciones al DataFrame
        result_df = df.copy()
        result_df['prediction'] = predictions.round(2)
        
        return result_df
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Modelo no encontrado. Ejecuta primero el entrenamiento. Detalles: {e}")
    except Exception as e:
        raise Exception(f"Error en predicción batch: {e}")

# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de predicción individual
    try:
        result = predict_single(
            temperature=22.5,
            demand_factor=0.75,
            operational_efficiency=0.85,
            energy_price=85.0,
            hour=14,
            day_of_week=2,
            month=6,
            is_weekend=0
        )
        
        print("🔮 Predicción individual:")
        print(f"  Valor predicho: {result['prediction']}")
        print(f"  Modelo usado: {result['model_type']}")
        print(f"  MAE del modelo: {result['model_mae']:.2f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")