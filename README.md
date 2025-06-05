# Proyecto Modelado Predictivo de Condiciones Ambientales para Agricultura Urbana Vertical de Precisión con Fotoluminiscencia
# Bootcamp en Inteligencia Artificial
# Talentotech2
# Autor: Víctor C. Vladimir Cortés Arévalo
### E-Mail: vladimir.cortes@outlook.com 

📋 Descripción General
Este proyecto implementa un pipeline completo de Machine Learning para optimizar un sistema de agricultura urbana vertical en ambiente controlado que utiliza tecnología de fotoluminiscencia. El sistema analiza 26 parámetros ambientales, lumínicos y de control para predecir y optimizar la eficiencia fotosintética y la intensidad de fotoluminiscencia en cultivos verticales.
🎯 Objetivos del Proyecto

Predecir la eficiencia fotosintética basada en condiciones ambientales controladas
Optimizar la intensidad de fotoluminiscencia para maximizar el crecimiento de plantas
Identificar patrones temporales y condiciones óptimas para el cultivo vertical
Proporcionar insights para la optimización automática del sistema de control ambiental
Desarrollar modelos robustos para implementación en producción

🏗️ Estructura del Pipeline
1. 00_EDA_agricultura_vertical.py - Análisis Exploratorio de Datos

Análisis completo de 50,000 registros de sensores
Visualización de patrones temporales y estacionales
Análisis de correlaciones entre variables ambientales
Identificación de condiciones óptimas de crecimiento
Dashboard interactivo del sistema

2. 01_data_preprocessing.py - Preprocesamiento de Datos

Limpieza y validación de datos de sensores
Manejo de valores faltantes y outliers
Validación de rangos específicos para agricultura vertical
División estratificada de datos (train/validation/test)
Control de calidad de datos

3. 02_feature_engineering.py - Ingeniería de Características

Creación de features avanzadas basadas en conocimiento del dominio
Features de interacciones entre variables ambientales
Transformaciones temporales (ciclicidad, estacionalidad)
Selección de features usando múltiples métodos
Análisis de importancia por grupos de variables

4. 03_train_models.py - Entrenamiento de Modelos

Entrenamiento de múltiples algoritmos (RF, XGBoost, LightGBM, etc.)
Optimización de hiperparámetros con GridSearch
Modelos separados para eficiencia fotosintética y fotoluminiscencia
Validación cruzada y métricas de evaluación
Comparación sistemática de rendimiento

5. 04_model_evaluation.py - Evaluación Completa de Modelos

Análisis visual detallado de predicciones
Análisis de importancia de features (permutation, SHAP)
Evaluación de robustez ante ruido
Análisis de casos extremos y estabilidad
Preparación para deployment en producción

🔬 Variables del Sistema
Variables Ambientales

Temperatura (°C), Humedad relativa (%)
Concentración de CO₂ (ppm), Presión de vapor (kPa)
Calidad del aire (PM2.5, VOCs, NO₂, O₃, AQI)

Variables Lumínicas

PAR (μmol/m²/s), PPFD (μmol/m²/s)
Espectro lumínico (nm), Intensidad fotoluminiscente

Variables de Control

Ajustes automáticos de temperatura, humedad, CO₂ y PAR
Porcentaje de purificación requerida
Condiciones categorizadas del ambiente

Variables Target

Eficiencia fotosintética (%) - Variable principal de optimización
Intensidad de fotoluminiscencia - Variable de control del sistema lumínico

🛠️ Tecnologías Utilizadas
Core ML/Data Science

Python 3.8+
pandas, numpy - Manipulación de datos
scikit-learn - Machine Learning
xgboost, lightgbm - Gradient boosting
scipy, statsmodels - Análisis estadístico

Visualización y Análisis

matplotlib, seaborn - Visualizaciones
shap - Interpretabilidad de modelos
plotly - Dashboards interactivos

Procesamiento y Deployment

joblib - Serialización de modelos
json - Configuración y metadatos

🚀 Cómo Ejecutar el Pipeline
Prerrequisitos
bashpip install pandas numpy scikit-learn matplotlib seaborn
pip install xgboost lightgbm shap scipy
Ejecución Secuencial
bash# 1. Análisis Exploratorio
python 00_EDA_agricultura_vertical.py

# 2. Preprocesamiento
python 01_data_preprocessing.py

# 3. Feature Engineering
python 02_feature_engineering.py

# 4. Entrenamiento de Modelos
python 03_train_models.py

# 5. Evaluación Final
python 04_model_evaluation.py
📁 Estructura de Directorios
proyecto/
├── data/
│   ├── raw/                    # Datos originales de sensores
│   └── processed/              # Datos procesados y features
├── models/                     # Modelos entrenados y metadatos
├── notebooks/                  # Análisis exploratorios adicionales
├── scripts/
│   ├── 00_EDA_agricultura_vertical.py
│   ├── 01_data_preprocessing.py
│   ├── 02_feature_engineering.py
│   ├── 03_train_models.py
│   └── 04_model_evaluation.py
└── README.md
📊 Resultados Esperados

Precisión: R² > 0.8 para ambos targets
Robustez: Degradación < 5% con ruido del 10%
Interpretabilidad: Top features identificadas para optimización
Producción: Modelos listos para deployment con configuración completa

🎯 Aplicaciones Prácticas
Optimización del Sistema

Control automático de condiciones ambientales
Alertas predictivas para mantenimiento
Optimización energética del sistema de iluminación

Insights Agronómicos

Identificación de condiciones óptimas por tipo de cultivo
Patrones estacionales y circadianos
Relaciones entre calidad del aire y productividad

Escalabilidad

Transferencia a otros cultivos verticales
Adaptación a diferentes ubicaciones geográficas
Integración con IoT y sistemas de control

📈 Métricas de Rendimiento

MAE (Mean Absolute Error): Error promedio absoluto
R² (Coefficient of Determination): Varianza explicada
MAPE (Mean Absolute Percentage Error): Error porcentual
Robustez: Estabilidad ante perturbaciones
Interpretabilidad: Explicabilidad de decisiones

🔄 Mantenimiento y Mejoras
Monitoreo Continuo

Validación de rangos de entrada
Detección de drift en datos
Alertas de degradación de performance

Reentrenamiento

Pipeline automatizado para nuevos datos
Validación A/B de modelos actualizados
Historial de versiones de modelos


Proyecto desarrollado para la optimización de sistemas de agricultura vertical urbana con tecnología de fotoluminiscencia avanzada.