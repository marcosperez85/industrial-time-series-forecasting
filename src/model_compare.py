import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def load_and_prepare(path="../data/processed/industrial_timeseries_featured.csv"):
    """
    Carga y prepara los datos para comparación de modelos.
    
    Args:
        path (str): Ruta al dataset procesado
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    
    # Features para el modelo
    feature_columns = [
        'temperature', 'demand_factor', 'operational_efficiency', 'energy_price',
        'hour', 'day_of_week', 'month', 'is_weekend',
        'lag_1h', 'lag_24h', 'rolling_mean_24h', 'rolling_std_24h',
        'temp_squared', 'demand_efficiency_interaction'
    ]
    
    X = df[feature_columns]
    y = df["value"]
    
    # Split temporal (sin shuffle para mantener orden temporal)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test

def calculate_mape(y_true, y_pred):
    """Calcula Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def compare(save_models=False):
    """
    Compara diferentes modelos de regresión y muestra sus métricas.
    
    Args:
        save_models (bool): Si es True, guarda los modelos entrenados.
    """
    print("🔄 Cargando y preparando datos...")
    X_train, X_test, y_train, y_test = load_and_prepare()
    
    print(f"Datos de entrenamiento: {len(X_train):,}")
    print(f"Datos de prueba: {len(X_test):,}")
    print(f"Features: {len(X_train.columns)}")

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = []
    trained_models = {}

    print("\n🔄 Entrenando y evaluando modelos...")
    
    for name, model in models.items():
        print(f"  - Entrenando {name}...")
        
        # Entrenar modelo
        model.fit(X_train, y_train)
        
        # Predicciones
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Métricas
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mape = calculate_mape(y_test, y_pred_test)

        results.append({
            "Model": name,
            "Train_MAE": train_mae,
            "Test_MAE": test_mae,
            "Train_RMSE": train_rmse,
            "Test_RMSE": test_rmse,
            "Train_R2": train_r2,
            "Test_R2": test_r2,
            "Test_MAPE": test_mape
        })
        
        trained_models[name] = model
        
        # Guardar modelo solo si se solicita explícitamente
        if save_models:
            os.makedirs("../models", exist_ok=True)
            model_path = f"../models/{name.lower()}_model.pkl"
            
            if os.path.exists(model_path):
                print(f"    ⚠️  Sobreescribiendo modelo existente en {model_path}")
                
            joblib.dump(model, model_path)
            print(f"    ✅ Modelo {name} guardado en {model_path}")

    # Crear y mostrar tabla de resultados
    df_results = pd.DataFrame(results)
    
    print("\n📊 COMPARACIÓN DE MODELOS:")
    print("=" * 80)
    
    # Mostrar métricas formateadas
    display_df = df_results.copy()
    numeric_cols = ['Train_MAE', 'Test_MAE', 'Train_RMSE', 'Test_RMSE', 'Train_R2', 'Test_R2', 'Test_MAPE']
    for col in numeric_cols:
        if col in ['Train_R2', 'Test_R2']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        elif col == 'Test_MAPE':
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
        else:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
    
    print(display_df.to_string(index=False))
    
    # Identificar el mejor modelo según diferentes métricas
    best_mae = df_results.loc[df_results["Test_MAE"].idxmin()]
    best_r2 = df_results.loc[df_results["Test_R2"].idxmax()]
    best_mape = df_results.loc[df_results["Test_MAPE"].idxmin()]
    
    print(f"\n🏆 MEJORES MODELOS:")
    print(f"  • Menor MAE: {best_mae['Model']} (MAE: {best_mae['Test_MAE']:.2f})")
    print(f"  • Mayor R²: {best_r2['Model']} (R²: {best_r2['Test_R2']:.4f})")
    print(f"  • Menor MAPE: {best_mape['Model']} (MAPE: {best_mape['Test_MAPE']:.2f}%)")
    
    # Guardar el mejor modelo según MAE
    if save_models:
        best_model_name = best_mae['Model']
        best_model = trained_models[best_model_name]
        
        # Información del modelo
        feature_info = {
            'features': list(X_train.columns),
            'model_type': best_model_name,
            'test_mae': best_mae['Test_MAE'],
            'test_r2': best_mae['Test_R2'],
            'test_mape': best_mae['Test_MAPE']
        }
        
        joblib.dump(best_model, "../models/best_model.pkl")
        joblib.dump(feature_info, "../models/model_info.pkl")
        
        print(f"\n✅ Mejor modelo guardado como:")
        print(f"  • ../models/best_model.pkl")
        print(f"  • ../models/model_info.pkl")
    
    return df_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compara diferentes modelos de regresión para series temporales industriales")
    parser.add_argument("--save", action="store_true", 
                       help="Guardar los modelos entrenados (por defecto: no guardar)")
    
    args = parser.parse_args()
    
    try:
        results = compare(save_models=args.save)
        print(f"\n✅ Comparación completada exitosamente")
    except FileNotFoundError as e:
        print(f"❌ Error: Archivo no encontrado. Asegúrate de haber ejecutado primero el notebook 03_feature_engineering.ipynb")
        print(f"Detalles: {e}")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")