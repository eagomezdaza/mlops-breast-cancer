#!/bin/bash
echo "🔧 CREANDO PROYECTO MLOps COMPLETO..."

# Crear directorio principal
mkdir -p mlops-breast-cancer
cd mlops-breast-cancer

# Crear estructura de directorios
mkdir -p app data models tests .github/workflows

# Crear requirements.txt
cat > app/requirements.txt << 'REQ_EOF'
flask==2.3.3
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
joblib==1.3.2
gunicorn==21.2.0
REQ_EOF

# Crear create_dataset.py
cat > create_dataset.py << 'DS_EOF'
import pandas as pd
from sklearn.datasets import load_breast_cancer
import os

print("Creando dataset...")
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target
df['diagnosis'] = df['diagnosis'].map({0: 'M', 1: 'B'})
df['id'] = range(1, len(df) + 1)
cols = ['id', 'diagnosis'] + [col for col in df.columns if col not in ['id', 'diagnosis']]
df = df[cols]
os.makedirs('data', exist_ok=True)
df.to_csv('data/breast_cancer.csv', index=False)
print("Dataset creado en 'data/breast_cancer.csv'")
DS_EOF

# Crear train_model.py
cat > train_model.py << 'TRAIN_EOF'
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

df = pd.read_csv('data/breast_cancer.csv')
df = df.drop('id', axis=1)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/trained_model.pkl')
joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')
print("Modelo guardado en 'models/trained_model.pkl'")
TRAIN_EOF

# Crear app.py
cat > app/app.py << 'APP_EOF'
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
models_path = os.path.join(project_root, 'models')

try:
    model = joblib.load(os.path.join(models_path, 'trained_model.pkl'))
    EXPECTED_FEATURES = joblib.load(os.path.join(models_path, 'feature_names.pkl'))
except:
    model = None
    EXPECTED_FEATURES = []

@app.route('/')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'features_loaded': len(EXPECTED_FEATURES) > 0
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo no disponible'}), 503
    
    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({'error': 'Datos inválidos'}), 400
    
    features = data['features']
    if len(features) != len(EXPECTED_FEATURES):
        return jsonify({'error': f'Esperadas {len(EXPECTED_FEATURES)} características'}), 400
    
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)[0]
    probability = model.predict_proba(features_array)[0]
    
    return jsonify({
        'prediction': int(prediction),
        'probability_malignant': float(probability[1]),
        'probability_benign': float(probability[0]),
        'interpretation': 'Maligno' if prediction == 1 else 'Benigno'
    })

@app.route('/features')
def get_features():
    return jsonify({
        'expected_features': EXPECTED_FEATURES,
        'count': len(EXPECTED_FEATURES)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
APP_EOF

# Crear Dockerfile
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

# Crear archivos vacíos necesarios
touch app/__init__.py
touch tests/test_api.py
touch .dockerignore
touch .github/workflows/ci-cd.yml

# Crear README.md mejorado
cat > README.md << 'README_EOF'
# MLOps - Sistema de Predicción de Cáncer de Mama

## Descripción
Sistema completo MLOps para predicción de cáncer de mama usando Random Forest.

## Instalación Rápida

### 1. Instalar dependencias:
cd app
pip install -r requirements.txt

### 2. Crear dataset y entrenar modelo:
python create_dataset.py
python train_model.py

### 3. Ejecutar API:
cd app
python app.py

### 4. Probar:
curl http://localhost:5000/
curl http://localhost:5000/features

curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"features": [13.08,15.71,85.63,520.0,0.1075,0.127,0.04568,0.0311,0.1967,0.06811,0.1852,0.7477,1.383,14.67,0.004097,0.01898,0.01698,0.00649,0.01678,0.002425,14.5,20.49,96.09,630.5,0.1312,0.2776,0.189,0.07283,0.3184,0.08183]}'

## Docker
docker build -t mlops-breast-cancer .
docker run -p 5000:5000 mlops-breast-cancer

## Repositorios
- GitHub: https://github.com/eagomezdaza/mlops-breast-cancer
- Docker Hub: https://hub.docker.com/r/johnegomez/mlops-breast-cancer
README_EOF

echo "🎉 PROYECTO CREADO EXITOSAMENTE!"
echo "📁 Directorio: mlops-breast-cancer"
echo ""
echo "🚀 PARA EJECUTAR:"
echo "1. cd mlops-breast-cancer"
echo "2. cd app && pip install -r requirements.txt"
echo "3. cd .. && python create_dataset.py"
echo "4. python train_model.py"
echo "5. cd app && python app.py"
echo ""
echo "🌐 Probar: curl http://localhost:5000/"
