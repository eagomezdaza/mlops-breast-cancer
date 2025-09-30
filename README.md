# MLOps — Sistema de Predicción de Cáncer de Mama (Flask + Docker)

Sistema completo para entrenar un modelo (Breast Cancer Wisconsin), exponerlo como **API REST** con **Flask**, contenedorizado con **Docker** y con pruebas de humo.

## 📦 Arquitectura (resumen)
- **Entrenamiento**: `sklearn.datasets.load_breast_cancer` → `RandomForestClassifier` → guarda `src/model/breast_cancer_model.pkl` y `src/model/model_info.pkl`. 
- **API Flask**: endpoints `GET /`, `GET /health`, `POST /predict`.
- **Contenedor**: `docker/Dockerfile`.
- **Pruebas**: `tests/test_api.py` (health, home y predicción).
- **Makefile**: atajos (train, run, test, docker-build, docker-run).

## 📁 Estructura del repo
```
mlops-breast-cancer/
├── app/
│   ├── app.py
│   ├── model.py
│   └── requirements.txt
├── data/
│   └── breast_cancer.csv              # opcional (si usas CSV propio)
├── models/
│   └── trained_model.pkl              # se genera al entrenar
├── tests/
│   ├── test_api.py
│   └── test_model.py
├── Dockerfile
├── .dockerignore
├── .github/
│   └── workflows/
│       └── ci-cd.yml                  # workflow CI/CD
├── train_model.py
└── README.md
```

## ✅ Requisitos
- Python 3.11 (se recomienda usar entorno virtual `venv`)
- Docker (opcional, para levantar la API en contenedor)

## 🚀 Instalación (local)
Clona el repositorio y prepara el entorno:

```bash
git clone https://github.com/eagomezdaza/mlops-breast-cancer.git
cd mlops-breast-cancer
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r app/requirements.txt
```

## 🧠 Entrenamiento del modelo
Ejecuta el script de entrenamiento para generar el modelo serializado:

```bash
python train_model.py
```

Esto entrena un **RandomForestClassifier** usando el dataset *Breast Cancer Wisconsin* incluido en `scikit-learn`.  
Al finalizar se guardará el archivo del modelo (ej. `models/trained_model.pkl`) junto con información del entrenamiento (accuracy, features, etc.).

Ejemplo de salida:
```
🚀 Iniciando entrenamiento del modelo...
✅ Modelo guardado. Accuracy=0.9561
```

## 🌐 Ejecutar la API Flask
Levanta el servicio de predicción:

```bash
python app/app.py
```

La API quedará corriendo en [http://localhost:8000](http://localhost:8000).

### Endpoints disponibles
- `GET /` → estado general del servicio  
- `GET /health` → estado del modelo (`accuracy`, carga, etc.)  
- `POST /predict` → recibe un JSON con 30 features y entrega la predicción

Ejemplo de request:
```bash
curl -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d '{"features":[17.99,10.38,122.8,1001.0,...,0.1189]}'
```

Ejemplo de respuesta:
```json
{
  "prediction": "benign",
  "confidence": 0.97
}
```

## 🧪 Pruebas (Testing)
Ejecuta las pruebas de humo incluidas:

```bash
make test
```

Salida esperada:
```
🧪 Probando la aplicación...
health: 200 {...}
home: 200 {...}
predict: 200 {"prediction": "benign", "confidence": 0.97}
```

## 🐳 Dockerización
Construir la imagen:

```bash
make docker-build
```

Ejecutar el contenedor (puerto 8000 o libre en host):

```bash
make docker-run PORT=8000
```

Probar con `curl`:
```bash
curl http://localhost:8000/health
```

## ⚙️ CI/CD con GitHub Actions
Este repo incluye un flujo de trabajo en `.github/workflows/ci-cd.yml` que automatiza:

- Instalación de dependencias  
- Entrenamiento del modelo  
- Levantar API Flask en background  
- Ejecutar pruebas de humo  

Badge de estado:
![CI](https://github.com/eagomezdaza/mlops-breast-cancer/actions/workflows/ci-cd.yml/badge.svg)

## 📑 Documentación y Entrega
El proyecto se entrega con:
- Código fuente organizado  
- Archivo `README.md` documentado  
- Reporte PDF con capturas de pantalla, explicación del flujo y reflexiones sobre buenas prácticas  

---
✍️ Autor: [@eagomezdaza](https://github.com/eagomezdaza)
