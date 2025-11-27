# 🏭 Industrial Time Series Forecasting

Pipeline completo y modular para predecir valores futuros en series temporales industriales. Diseñado como un **template reutilizable** para experimentación rápida, benchmarking de modelos y despliegue de prototipos, completamente **configurable a través de un solo archivo YAML**.

## 🎯 Objetivo

Desarrollar un sistema de predicción **genérico y configurable** para series temporales industriales que permita:

- **Adaptación rápida a nuevos datasets** editando solo `config.yaml`
- Experimentación ágil con diferentes algoritmos de ML
- Feature engineering avanzado y automatizado
- Comparación objetiva y reproducible de modelos
- Despliegue inmediato a través de API REST
- Visualización y análisis automático de resultados

## ✨ Características

### 🔧 **Modularización Completa**
- **Un solo archivo de configuración**: `config.yaml` controla todo el pipeline
- Código reutilizable y sin valores hardcodeados
- Cambio de dataset sin modificar código fuente

### 🤖 **Machine Learning Avanzado**
- **Múltiples algoritmos**: Linear Regression, Random Forest, Gradient Boosting
- **Feature Engineering automático**: lags, rolling statistics, interacciones, términos cuadráticos
- **Comparación automática** de modelos con múltiples métricas

### 📊 **Análisis y Visualización**
- **EDA completamente genérico** que se adapta a cualquier dataset
- Visualizaciones automáticas de patrones temporales
- Análisis de correlaciones y feature importance

### 🚀 **Despliegue y Producción**
- **API REST con FastAPI** para predicciones en tiempo real
- Scripts CLI para entrenamiento automatizado
- Organización estilo MLOps para escalabilidad

### 🔄 **Reproducibilidad**
- Pipeline determinista y versionado
- Notebooks estructurados y modulares
- Métricas completas para evaluación objetiva

## 🏗️ Arquitectura del Proyecto

```
industrial-time-series-forecasting/
│
├── config.yaml                           # 🎛️  CONFIGURACIÓN CENTRAL
│
├── data/                                 # 📊 Datos del proyecto
│   ├── raw/                             #     Datos originales
│   │   └── industrial_timeseries.csv   #     Dataset generado/importado
│   └── processed/                       #     Datos con feature engineering
│       └── industrial_timeseries_featured.csv
│
├── models/                              # 🤖 Modelos entrenados
│   ├── best_model.pkl                   #     Mejor modelo seleccionado
│   ├── model_info.pkl                  #     Metadatos y métricas
│   ├── linear_regression_model.pkl     #     Modelos individuales
│   ├── random_forest_model.pkl         #
│   └── gradient_boosting_model.pkl     #
│
├── notebooks/                           # 📓 Análisis y experimentación
│   ├── 01_load.ipynb                   #     Carga y validación de datos
│   ├── 02_eda.ipynb                    #     Análisis exploratorio genérico
│   ├── 03_feature_engineering.ipynb    #     Creación de características
│   ├── 04_model.ipynb                  #     Entrenamiento y comparación
│   └── 05_forecast.ipynb               #     Predicciones futuras
│
├── src/                                 # 🔧 Código fuente modular
│   ├── __init__.py                     #
│   ├── config_loader.py                #     Carga de configuración YAML
│   ├── data_loader.py                  #     Funciones de carga de datos
│   ├── create_dataset.py               #     Generador de datos sintéticos
│   ├── train_model.py                  #     Entrenamiento individual
│   ├── model_compare.py                #     Comparación de modelos
│   ├── predict.py                      #     Sistema de predicciones
│   ├── main_api.py                     #     API REST con FastAPI
│   └── features/                       #     Motor de feature engineering
│       ├── __init__.py                 #
│       └── feature_engineering.py      #     FeatureEngineeringEngine
│
├── requirements.txt                     # 📦 Dependencias del proyecto
└── README.md                           # 📖 Documentación principal
```

## 🚀 Instalación y Setup

### 1. Clonar el repositorio
```bash
git clone <repository-url>
cd industrial-time-series-forecasting
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Generar dataset de ejemplo
```bash
cd src
python create_dataset.py
```

### 4. Verificar instalación
```bash
cd src
python load.py
```

## ⚙️ Configuración Central - config.yaml

Todo el comportamiento del pipeline se controla desde `config.yaml`:

```yaml
dataset:
  path: "data/raw/industrial_timeseries.csv"
  datetime_col: "timestamp"
  target_col: "value"
  freq: "H"
  raw_feature_columns:
    - temperature
    - demand_factor
    - operational_efficiency
    - energy_price

feature_engineering:
  time_features: true
  lags: [1, 24, 48, 168]
  rolling:
    windows: [24]
    functions: ["mean", "std"]
  interactions:
    - ["demand_factor", "operational_efficiency"]
    - ["temperature", "demand_factor"]
  squared_terms:
    - "temperature"

training:
  test_ratio: 0.2
  models:
    random_forest:
      n_estimators: 300
      max_depth: 10
    gradient_boosting:
      n_estimators: 300
      learning_rate: 0.05
```

### 🔄 Cambiar de Dataset

Para usar un **nuevo dataset**, solo necesitas:

1. **Colocar tu CSV** en `data/raw/`
2. **Actualizar config.yaml**:
   ```yaml
   dataset:
     path: "data/raw/tu_dataset.csv"
     datetime_col: "fecha"           # Tu columna de tiempo
     target_col: "ventas"            # Tu variable objetivo
     raw_feature_columns:            # Tus features
       - precio
       - inventario  
       - promocion
   ```
3. **Ejecutar el pipeline** - todo se adapta automáticamente

## 📓 Uso de Notebooks

### Flujo Recomendado

```bash
# 1. Carga y validación inicial
jupyter notebook notebooks/01_load.ipynb

# 2. Análisis exploratorio automático  
jupyter notebook notebooks/02_eda.ipynb

# 3. Feature engineering configurable
jupyter notebook notebooks/03_feature_engineering.ipynb

# 4. Entrenamiento y comparación de modelos
jupyter notebook notebooks/04_model.ipynb

# 5. Generación de predicciones futuras
jupyter notebook notebooks/05_forecast.ipynb
```

### Características de los Notebooks

- **Completamente genéricos**: Se adaptan automáticamente al dataset configurado
- **Sin código hardcodeado**: Todas las configuraciones vienen de `config.yaml`
- **Análisis automático**: EDA detecta tipos de datos y genera visualizaciones apropiadas
- **Reproducibles**: Mismos resultados en diferentes ejecuciones

## 🔧 Scripts de Línea de Comandos

### Entrenamiento Individual
```bash
cd src

# Entrenar Random Forest (recomendado)
python train_model.py --model random_forest

# Entrenar Gradient Boosting  
python train_model.py --model gradient_boosting

# Entrenar Linear Regression
python train_model.py --model linear_regression
```

### Comparación Automática de Modelos
```bash
cd src

# Solo comparar rendimiento
python model_compare.py

# Comparar y guardar mejor modelo
python model_compare.py --save
```

### Validación de Configuración
```bash
cd src

# Verificar que config.yaml y datos son válidos
python load.py
```

## 🚀 API REST para Producción

### Iniciar servidor
```bash
cd src
python main_api.py
```

### Realizar predicciones
```bash
# Predicción individual
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "temperature": 22.5,
  "demand_factor": 0.75,
  "operational_efficiency": 0.85,
  "energy_price": 85.0,
  "hour": 14,
  "day_of_week": 2,
  "month": 6,
  "is_weekend": 0
}'

# Obtener información del modelo
curl http://localhost:8000/model-info

# Health check
curl http://localhost:8000/health
```

### Endpoints Disponibles

- `GET /` - Información general de la API
- `POST /predict` - Realizar predicción individual
- `GET /health` - Estado del servicio
- `GET /model-info` - Información del modelo cargado
- `GET /features` - Features requeridas por el modelo
- `GET /config` - Configuración actual del sistema
- `GET /docs` - Documentación interactiva (Swagger UI)

## 📊 Métricas de Evaluación

El sistema utiliza **múltiples métricas** para evaluación exhaustiva:

### Métricas Principales
- **MAE** (Mean Absolute Error): Error promedio absoluto
- **RMSE** (Root Mean Square Error): Penaliza errores grandes
- **R²** (R-squared): Proporción de varianza explicada
- **MAPE** (Mean Absolute Percentage Error): Error porcentual promedio

### Análisis Automático
- **Detección de Overfitting**: Comparación automática entre train/test
- **Feature Importance**: Ranking de variables más predictivas  
- **Correlación con Target**: Identificación de relaciones lineales
- **Análisis Temporal**: Patrones por hora, día, mes, estacionalidad

### Ejemplo de Salida
```
📊 COMPARACIÓN DE MODELOS:
================================================================================
Model               Train_MAE  Test_MAE  Train_R2  Test_R2   Test_RMSE  Test_MAPE
Linear Regression   15.23      18.45     0.8234    0.7891    24.67      2.34%
Random Forest       8.12       16.78     0.9456    0.8123    22.34      2.12% 
Gradient Boosting   6.89       15.23     0.9621    0.8345    20.45      1.98%

🏆 MEJORES MODELOS:
• Menor MAE: Gradient Boosting (MAE: 15.23)
• Mayor R²: Gradient Boosting (R²: 0.8345)  
• Menor MAPE: Gradient Boosting (MAPE: 1.98%)
```

## 🔍 Casos de Uso

### Para Data Scientists
- **Prototipado rápido** de modelos de forecasting
- **Benchmarking** de algoritmos en nuevos datasets
- **Feature engineering** sistemático y reproducible

### Para Ingenieros ML
- **Template base** para proyectos de series temporales
- **API lista para producción** con validación automática
- **Pipeline CI/CD** compatible con herramientas estándar

### Para Analistas de Negocio
- **Predicciones automáticas** sin conocimiento técnico profundo
- **Dashboards** y visualizaciones auto-generadas
- **Interpretabilidad** de modelos y features

## 🛠️ Personalización Avanzada

### Agregar Nuevos Modelos
```python
# En model_compare.py o train_model.py
from sklearn.svm import SVR

models['SVM'] = SVR(kernel='rbf', C=1.0)
```

### Nuevas Features de Ingeniería
```yaml
# En config.yaml
feature_engineering:
  lags: [1, 6, 12, 24, 168]  # Agregar más lags
  rolling:
    windows: [6, 12, 24, 168]  # Múltiples ventanas
    functions: ["mean", "std", "min", "max"]  # Más estadísticas
```

### Configurar para Nuevos Dominios
```yaml
# Ejemplo: Ventas de retail
dataset:
  path: "data/raw/retail_sales.csv"
  datetime_col: "date"
  target_col: "sales"
  raw_feature_columns:
    - price
    - inventory
    - promotion
    - competitor_price
```

## 📋 Dependencias Principales

```txt
pandas>=1.5.0          # Manipulación de datos
numpy>=1.21.0          # Cálculos numéricos  
scikit-learn>=1.1.0    # Machine learning
matplotlib>=3.5.0      # Visualización
seaborn>=0.11.0        # Visualización estadística
pyyaml>=6.0           # Configuración YAML
fastapi>=0.85.0       # API REST
joblib>=1.1.0         # Serialización de modelos
jupyter>=1.0.0        # Notebooks interactivos
```

## 🤝 Contribución

1. Fork del proyecto
2. Crear branch para nueva feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit de cambios (`git commit -m 'Agregar nueva característica'`)
4. Push a la branch (`git push origin feature/nueva-caracteristica`)
5. Crear Pull Request

## 📝 Próximas Mejoras

- [ ] **Modelos de Deep Learning** (LSTM, GRU, Transformer)
- [ ] **AutoML** para selección automática de hiperparámetros  
- [ ] **Detección de anomalías** en tiempo real
- [ ] **Dashboard interactivo** con Streamlit/Dash
- [ ] **Containerización** con Docker
- [ ] **Monitoreo de deriva** de datos y modelos
- [ ] **Explicabilidad** con SHAP/LIME
- [ ] **Pipeline de CI/CD** completo

---

## 🎯 Quick Start (5 minutos)

```bash
# 1. Clonar e instalar
git clone <repo>
cd industrial-time-series-forecasting  
pip install -r requirements.txt

# 2. Generar datos de ejemplo
cd src && python create_dataset.py

# 3. Ejecutar pipeline completo
jupyter notebook notebooks/01_load.ipynb      # Verificar datos
jupyter notebook notebooks/03_feature_engineering.ipynb  # Procesar  
jupyter notebook notebooks/04_model.ipynb     # Entrenar
python main_api.py                            # API REST

# 4. Tu sistema está listo! 🚀
```

**¿Preguntas?** Revisa los notebooks de ejemplo o abre un issue en el repositorio.