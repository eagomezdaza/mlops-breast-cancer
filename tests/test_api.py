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
