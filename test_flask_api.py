import requests
import json
from datetime import datetime

# Test Flask API
url = 'http://localhost:5000/api/predict'
data = {
    'latitude': 16.5,
    'longitude': 80.6,
    'timestamp': datetime.now().isoformat()
}

try:
    response = requests.post(url, json=data, timeout=5)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    if response.ok:
        result = response.json()
        print(f"Prediction: {result.get('prediction')}")
        print(f"Confidence: {result.get('confidence')}")
except Exception as e:
    print(f"Error: {e}")
