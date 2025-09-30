import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
import os

print("Creando dataset de Breast Cancer desde scikit-learn...")

# Cargar el dataset de breast cancer incluido en scikit-learn
data = load_breast_cancer()

# Crear DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Añadir la columna target (diagnosis)
df['diagnosis'] = data.target

# Mapear 0=Maligno, 1=Benigno (para coincidir con el dataset original de Kaggle) 
df['diagnosis'] = df['diagnosis'].map({0: 'M', 1: 'B'})

# Añadir columna id
df['id'] = range(1, len(df) + 1)

# Reordenar columnas para que id y diagnosis sean las primeras
cols = ['id', 'diagnosis'] + [col for col in df.columns if col not in ['id', 'diagnosis']]
df = df[cols]

# Guardar como CSV
os.makedirs('data', exist_ok=True)
df.to_csv('data/breast_cancer.csv', index=False)

print("Dataset creado exitosamente en 'data/breast_cancer.csv'")
print(f"Tamaño del dataset: {df.shape}")
print("Primeras 5 filas:")
print(df.head())
print(f"Distribución de diagnóstico: {df['diagnosis'].value_counts().to_dict()}")
