# Fraud Detection Pipeline

Este repositorio contiene un pipeline de detección de fraudes utilizando los datos proporcionados por la competencia **IEEE-CIS Fraud Detection** en Kaggle. El objetivo principal de este proyecto es construir un modelo que pueda detectar fraudes de manera efectiva, con un enfoque en maximizar el **recall**, dado que es más importante identificar todos los fraudes posibles que minimizar los falsos negativos. Sin embargo, también se cuidó que el **accuracy** no disminuyera significativamente para mantener un modelo balanceado y robusto.

## Descripción del Proyecto

En este proyecto, me enfoqué en la maximización del **recall** para asegurar que los fraudes se identificaran de manera efectiva, en línea con las reglas de negocio que priorizan detectar la mayor cantidad posible de fraudes. A diferencia de otros proyectos de la misma base de datos que se centran en maximizar la **accuracy**, decidí priorizar el recall debido a que, tras analizar proyectos similares, noté que muchos de ellos sacrifican la capacidad de detectar fraudes a favor de mejorar la precisión general del modelo, lo que podría resultar en más fraudes no detectados.

### Objetivos:
- **Maximizar el recall** sin perder demasiado en términos de accuracy.
- **Preprocesar los datos** para asegurar la limpieza, consistencia y calidad de los mismos antes de alimentar el modelo.
- **Crear un pipeline robusto**, que incluya el preprocesamiento de los datos y la implementación de un modelo de detección de fraudes, capaz de recibir nuevos datos y realizar predicciones de forma automática.

### Detalles del Pipeline
El pipeline final incluye:
1. **Preprocesamiento de los datos**: Limpieza, manejo de valores faltantes, normalización de características y transformación de las variables.
2. **Modelado**: Se implementó un modelo de aprendizaje automático optimizado para maximizar el recall. El modelo está listo para hacer predicciones con nuevos datos.
3. **Evaluación**: Se utilizaron métricas como recall, precisión, y F1-score para evaluar el rendimiento del modelo, asegurando que el modelo detecte la mayor cantidad de fraudes posible sin una gran penalización en accuracy.

El pipeline está estructurado de manera modular, lo que permite adaptarlo a otros conjuntos de datos o necesidades específicas. Además, está preparado para integrarse fácilmente en un entorno de producción.

## Conjunto de Datos

El conjunto de datos utilizado proviene de la competencia **IEEE-CIS Fraud Detection** en Kaggle. Los datos están disponibles para su descarga en la siguiente liga:

- [IEEE-CIS Fraud Detection - Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)

También puedes acceder a los datos directamente desde mi **Google Drive** en el siguiente enlace:

- [Descargar Datos - Google Drive](https://drive.google.com/drive/folders/11f--xFYxzVgFXF3MqtDaCBfbuF0Z10Ii?usp=sharing)

**Nota**: Los datos contienen información sobre transacciones y sus características, y se utilizan para construir un modelo que pueda predecir si una transacción es fraudulenta o no. El dataset incluye tanto variables numéricas como categóricas.

## Resultados

El mejor modelo es el pipeline_final_xgb.joblib, ha sido entrenado y evaluado utilizando diversas métricas, con un enfoque principal en maximizar el recall. Si bien el accuracy no se dejó de lado, el enfoque en la detección de fraudes se traduce en un modelo que prioriza los casos donde el fraude es más probable, y minimiza los falsos negativos.

## Uso del pipeline

Para poder hacer uso del pipeline final y realizar predicciones correctamente, se deben seguir los siguientes pasos:

```python

# 1.- Importar las clases y funciones personalizadas
from custom_pipeline import DropColumns, ThresholdClassifier
from joblib import load
import pandas as pd

# 2.- Cargar el pipeline final entrenado
pipeline_final_xgb = load('pipeline_final_xgb.joblib')

# 3.- Cargar los nuevos datos a predecir
# Nota: El DataFrame debe tener la misma estructura y nombres de columnas que los archivos utilizados durante el entrenamiento (train.csv)
nuevos_datos = pd.read_csv('nuevos_datos.csv')

# 4.- Realizar las predicciones
predicciones = pipeline_final_xgb.predict(nuevos_datos)

# 5.-  Mostrar los resultados
print(predicciones)

