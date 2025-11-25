import pandas as pd
import numpy as np
import argparse
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_and_prepare_data(data_path="../data/processed/industrial_timeseries_featured.csv"):
    """
    Carga y prepara los datos para el entrenamiento.
    
    Args:
        data_path (str): Ruta al dataset procesado
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print(f"🔄 Cargando datos desde: {data_path}")
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    
    # Features para el modelo
    feature_columns = [
        'temperature', 'demand_factor', 'operational_efficiency', 'energy_price',
        'hour', 'day_of_week', 'month', 'is_weekend',
        'lag_1h', 'lag_24h', 'rolling_mean_24h', 'rolling_std_24h',
        'temp_squared', 'demand_efficiency_interaction'
    ]
    
    print(f"📊 Dataset cargado: {len(df):,} registros")
    print(f"🔧 Features seleccionadas: {len(feature_columns)}")
    
    X = df[feature_columns]
    y = df["value"]
    
    # Split temporal (80% entrenamiento, 20% prueba)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"📈 Datos de entrenamiento: {len(X_train):,}")
    print(f"🧪 Datos de prueba: {len(X_test):,}")
    
    return X_train, X_test, y_train, y_test, feature_columns

def calculate_mape(y_true, y_pred):
    """Calcula Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def train(model_type="random_forest", save_model=True):
    """
    Entrena un modelo de predicción para series temporales industriales.
    
    Args:
        model_type (str): Tipo de modelo a entrenar
        save_model (bool): Si guardar el modelo entrenado
    """
    try:
        # Cargar y preparar datos
        X_train, X_test, y_train, y_test, feature_columns = load_and_prepare_data()
        
        # Seleccionar el tipo de modelo
        print(f"\n🤖 Configurando modelo: {model_type}")
        
        if model_type.lower() == "linear_regression":
            model = LinearRegression()
            model_filename = "../models/linear_regression_model.pkl"
        elif model_type.lower() == "gradient_boosting":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model_filename = "../models/gradient_boosting_model.pkl"
        else:  # default to random_forest
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model_filename = "../models/random_forest_model.pkl"
        
        # Entrenar el modelo
        print("🔄 Entrenando modelo...")
        model.fit(X_train, y_train)
        
        # Realizar predicciones
        print("📊 Evaluando modelo...")
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calcular métricas
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mape = calculate_mape(y_test, y_pred_test)
        
        # Mostrar resultados
        print(f"\n📈 RESULTADOS DEL MODELO {model_type.upper()}:")
        print("=" * 50)
        print(f"  ENTRENAMIENTO:")
        print(f"    MAE:  {train_mae:.2f}")
        print(f"    RMSE: {train_rmse:.2f}")
        print(f"    R²:   {train_r2:.4f}")
        print(f"  ")
        print(f"  PRUEBA:")
        print(f"    MAE:  {test_mae:.2f}")
        print(f"    RMSE: {test_rmse:.2f}")
        print(f"    R²:   {test_r2:.4f}")
        print(f"    MAPE: {test_mape:.2f}%")
        
        # Verificar overfitting
        overfitting_mae = ((train_mae - test_mae) / test_mae) * 100
        overfitting_r2 = ((train_r2 - test_r2) / test_r2) * 100
        
        print(f"\n🔍 ANÁLISIS DE OVERFITTING:")
        print(f"    Diferencia MAE: {overfitting_mae:+.1f}%")
        print(f"    Diferencia R²: {overfitting_r2:+.1f}%")
        
        if abs(overfitting_mae) > 10 or abs(overfitting_r2) > 10:
            print("    ⚠️  Posible overfitting detectado")
        else:
            print("    ✅ Modelo generaliza bien")
        
        # Guardar el modelo
        if save_model:
            os.makedirs("../models", exist_ok=True)
            
            # Guardar modelo
            joblib.dump(model, model_filename)
            
            # Guardar información del modelo
            model_info = {
                'features': feature_columns,
                'model_type': model_type,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'test_mape': test_mape,
                'train_mae': train_mae,
                'train_r2': train_r2
            }
            
            info_filename = model_filename.replace('.pkl', '_info.pkl')
            joblib.dump(model_info, info_filename)
            
            print(f"\n💾 MODELO GUARDADO:")
            print(f"    Modelo: {model_filename}")
            print(f"    Info: {info_filename}")
        
        return model, {
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mape': test_mape
        }
        
    except FileNotFoundError as e:
        print(f"❌ Error: Archivo no encontrado.")
        print(f"   Asegúrate de haber ejecutado primero el notebook 03_feature_engineering.ipynb")
        print(f"   Detalles: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena un modelo para predicción de series temporales industriales")
    parser.add_argument("--model", type=str, default="random_forest",
                      choices=["linear_regression", "random_forest", "gradient_boosting"],
                      help="Tipo de modelo a entrenar")
    parser.add_argument("--no-save", action="store_true",
                      help="No guardar el modelo entrenado")
    
    args = parser.parse_args()
    
    print("🚀 ENTRENAMIENTO DE MODELO - SERIES TEMPORALES INDUSTRIALES")
    print("=" * 60)
    
    model, metrics = train(model_type=args.model, save_model=not args.no_save)
    
    if model is not None:
        print(f"\n✅ Entrenamiento completado exitosamente!")
    else:
        print(f"\n❌ Entrenamiento falló.")