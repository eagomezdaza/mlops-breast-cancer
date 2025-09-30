#!/bin/bash

# Script para crear proyecto MLOps completo
# Ejecutar: bash setup_completo.sh

echo "🚀 Creando proyecto MLOps completo..."

# Crear estructura de directorios
mkdir -p app data models tests .github/workflows

# 1. Crear requirements.txt
cat > app/requirements.txt << 'REQ_EOF'
flask==2.3.3
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
joblib==1.3.2
gunicorn==21.2.0
REQ_EOF

# 2. Crear train_model.py
cat > train_model.py << 'TRAIN_EOF'
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
    
    # Limpieza y preparación
    df = df.drop('id', axis=1)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    
    return X, y

def train_model():
    """Entrena y guarda el modelo"""
    print("Cargando datos...")
    X, y = load_and_prepare_data()
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Entrenar modelo
    print("Entrenando modelo...")
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
    print("Modelo guardado en 'models/trained_model.pkl'")
    
    return model, X.columns.tolist()

if __name__ == "__main__":
    train_model()
TRAIN_EOF

# 3. Crear app.py
cat > app/app.py << 'APP_EOF'
from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
import os

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Cargar modelo al iniciar
try:
    model = joblib.load('models/trained_model.pkl')
    logger.info("Modelo cargado exitosamente")
except Exception as e:
    logger.error(f"Error cargando el modelo: {e}")
    model = None

# Características esperadas
EXPECTED_FEATURES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'
]

@app.route('/', methods=['GET'])
def health_check():
    """Endpoint para verificar el estado del servicio"""
    return jsonify({
        'status': 'healthy',
        'message': 'MLOps Breast Cancer API está funcionando',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para realizar predicciones"""
    try:
        if model is None:
            return jsonify({'error': 'Modelo no disponible'}), 503
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se proporcionaron datos JSON'}), 400
        
        if 'features' not in data:
            return jsonify({'error': 'Falta el campo "features"'}), 400
        
        features = data['features']
        
        if len(features) != len(EXPECTED_FEATURES):
            return jsonify({
                'error': f'Número incorrecto de características. Esperado: {len(EXPECTED_FEATURES)}, Recibido: {len(features)}'
            }), 400
        
        features_array = np.array(features).reshape(1, -1)
        
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        
        logger.info(f"Predicción realizada: {prediction}")
        
        return jsonify({
            'prediction': int(prediction),
            'probability_malignant': float(probability[1]),
            'probability_benign': float(probability[0]),
            'interpretation': 'Maligno' if prediction == 1 else 'Benigno'
        })
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@app.route('/features', methods=['GET'])
def get_expected_features():
    """Endpoint para obtener las características esperadas"""
    return jsonify({
        'expected_features': EXPECTED_FEATURES,
        'count': len(EXPECTED_FEATURES)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
APP_EOF

# 4. Crear Dockerfile
cat > Dockerfile << 'DOCKER_EOF'
FROM python:3.9-slim
WORKDIR /app
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ .
COPY models/ ./models/
RUN useradd -m -r appuser && chown -R appuser /app
USER appuser
EXPOSE 5000
CMD ["python", "app.py"]
DOCKER_EOF

# 5. Crear tests
cat > tests/test_api.py << 'TEST_EOF'
import unittest
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))
from app import app

class TestAPI(unittest.TestCase):
    
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    
    def test_health_check(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)
    
    def test_features_endpoint(self):
        response = self.app.get('/features')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('expected_features', data)

if __name__ == '__main__':
    unittest.main()
TEST_EOF

# 6. Crear CI/CD
cat > .github/workflows/ci-cd.yml << 'CI_EOF'
name: MLOps CI/CD Pipeline
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        cd app
        pip install -r requirements.txt
        pip install pytest
    - name: Run tests
      run: |
        python -m pytest tests/ -v
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Build Docker image
      run: |
        docker build -t mlops-breast-cancer:latest .
CI_EOF

# 7. Crear archivos vacíos necesarios
touch app/__init__.py
touch app/model.py
touch tests/test_model.py
touch .dockerignore

# 8. Crear README.md
cat > README.md << 'README_EOF'
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
README_EOF

echo "✅ Proyecto creado exitosamente!"
echo "📁 Estructura creada:"
find . -type f -name "*.py" -o -name "*.txt" -o -name "*.yml" -o -name "Dockerfile" -o -name "README.md" | sort

echo -e "\n📝 Próximos pasos:"
echo "1. Descargar dataset de Kaggle a data/breast_cancer.csv"
echo "2. Ejecutar: cd app && pip install -r requirements.txt"
echo "3. Ejecutar: python train_model.py"
echo "4. Ejecutar: python app.py"
