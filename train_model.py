import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def load_and_prepare_data():
    """Carga y prepara el dataset de Breast Cancer"""
    df = pd.read_csv('data/breast_cancer.csv')
    
    # El dataset de scikit-learn ya está limpio, solo necesitamos preparar las variables
    # Eliminar columnas no necesarias
    df = df.drop('id', axis=1)
    
    # Convertir diagnosis a numérico (M=1, B=0)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # Separar características y target
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    
    print(f"Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    return X, y

def train_model():
    """Entrena y guarda el modelo"""
    print("Cargando datos...")
    X, y = load_and_prepare_data()
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Datos de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Datos de prueba: {X_test.shape[0]} muestras")
    
    # Entrenar modelo
    print("Entrenando modelo Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluar modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy del modelo: {accuracy:.4f}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Guardar modelo
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/trained_model.pkl')
    
    # Guardar también los nombres de las características para la API
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    print("Modelo guardado en 'models/trained_model.pkl'")
    print("Características guardadas en 'models/feature_names.pkl'")
    print(f"Número de características: {len(feature_names)}")
    
    return model, feature_names

if __name__ == "__main__":
    train_model()
