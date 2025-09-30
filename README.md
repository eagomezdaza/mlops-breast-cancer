# MLOps - Sistema de Predicción de Cáncer de Mama

## Descripción
Sistema completo MLOps para predicción de cáncer de mama usando Random Forest, expuesto como API REST con Flask y contenedorizado con Docker.

## Estructura del Proyecto
mlops-breast-cancer/
├── app/
│   ├── __init__.py
│   ├── app.py
│   ├── model.py
│   └── requirements.txt
├── data/
│   └── breast_cancer.csv
├── models/
│   └── trained_model.pkl
├── tests/
│   ├── test_api.py
│   └── test_model.py
├── Dockerfile
├── .dockerignore
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── README.md
└── train_model.py

## Instalación y Uso

### 1. Descargar dataset
Descargar de: https://www.kaggle.com/datasets/ucimi/breast-cancer-wisconsin-data
Colocar en: data/breast_cancer.csv

### 2. Instalar dependencias
cd app
pip install -r requirements.txt

### 3. Entrenar modelo
python train_model.py

### 4. Ejecutar API
cd app
python app.py

### 5. Probar API
curl http://localhost:5000/
curl http://localhost:5000/features

curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"features": [17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871]}'

## Docker
docker build -t mlops-breast-cancer .
docker run -p 5000:5000 mlops-breast-cancer
