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

# Cargar modelo y características al iniciar
try:
    model = joblib.load('models/trained_model.pkl')
    EXPECTED_FEATURES = joblib.load('models/feature_names.pkl')
    logger.info("Modelo y características cargados exitosamente")
    logger.info(f"Número de características: {len(EXPECTED_FEATURES)}")
except Exception as e:
    logger.error(f"Error cargando el modelo o características: {e}")
    model = None
    EXPECTED_FEATURES = []

@app.route('/', methods=['GET'])
def health_check():
    """Endpoint para verificar el estado del servicio"""
    return jsonify({
        'status': 'healthy',
        'message': 'MLOps Breast Cancer API está funcionando',
        'model_loaded': model is not None,
        'features_loaded': len(EXPECTED_FEATURES) > 0
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para realizar predicciones"""
    try:
        # Validar que el modelo esté cargado
        if model is None:
            return jsonify({'error': 'Modelo no disponible'}), 503
        
        # Obtener y validar datos
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se proporcionaron datos JSON'}), 400
        
        # Validar características
        if 'features' not in data:
            return jsonify({'error': 'Falta el campo "features"'}), 400
        
        features = data['features']
        
        if len(features) != len(EXPECTED_FEATURES):
            return jsonify({
                'error': f'Número incorrecto de características. Esperado: {len(EXPECTED_FEATURES)}, Recibido: {len(features)}',
                'expected_count': len(EXPECTED_FEATURES)
            }), 400
        
        # Convertir a numpy array y predecir
        features_array = np.array(features).reshape(1, -1)
        
        # Realizar predicción
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        
        # Log de la predicción
        logger.info(f"Predicción realizada: {prediction}, Probabilidades: {probability}")
        
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
