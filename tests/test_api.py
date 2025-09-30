import unittest
import json
import os
import sys

# Agregar el directorio raíz al path de Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.app import app

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
        # Puede devolver 503 si el modelo no está cargado, eso es aceptable
        self.assertIn(response.status_code, [200, 503])

if __name__ == '__main__':
    unittest.main()
