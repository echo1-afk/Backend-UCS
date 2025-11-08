#!/usr/bin/env python3
"""
Enhanced Traffic Prediction API with Multiple Models
Supports LSTM, CNN-LSTM, and Ensemble predictions
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for models and scalers
models = {}
feature_scaler = None
target_scaler = None
model_metadata = {}

def load_all_models():
    """Load all available models (LSTM, CNN-LSTM, and best model)"""
    global models, feature_scaler, target_scaler, model_metadata
    
    try:
        print("\n" + "="*70)
        print("LOADING TRAFFIC PREDICTION MODELS")
        print("="*70)
        
        # Model paths to try
        model_configs = {
            'lstm': 'models/lstm_model.h5',
            'cnn_lstm': 'models/cnn_lstm_model.h5',
            'best': 'models/best_model.h5'
        }
        
        # Load each available model
        for model_name, model_path in model_configs.items():
            if os.path.exists(model_path):
                print(f"\n📥 Loading {model_name.upper()} model from: {model_path}")
                model = tf.keras.models.load_model(model_path, compile=False)
                model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
                models[model_name] = model
                print(f"✅ {model_name.upper()} model loaded successfully!")
            else:
                print(f"⚠️  {model_name.upper()} model not found at: {model_path}")
        
        if not models:
            raise FileNotFoundError("No models found! Please ensure model files exist in models/ directory")
        
        # Load scalers
        print("\n📥 Loading scalers...")
        feature_scaler = joblib.load('models/feature_scaler.pkl')
        target_scaler = joblib.load('models/target_scaler.pkl')
        print("✅ Scalers loaded successfully!")
        
        # Load metadata
        print("\n📥 Loading model metadata...")
        with open('models/model_metadata.json', 'r') as f:
            model_metadata = json.load(f)
        model_metadata['available_models'] = list(models.keys())
        print(f"✅ Metadata loaded! Available models: {', '.join(models.keys())}")
        
        print("\n" + "="*70)
        print(f"✅ LOADED {len(models)} MODEL(S) SUCCESSFULLY!")
        print("="*70 + "\n")
        
        return True
    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        return False

def preprocess_location_data(lat, lon, timestamp):
    """Preprocess location-based data for prediction"""
    dt = pd.to_datetime(timestamp)
    
    # Time-based features
    hour = dt.hour
    dow = dt.dayofweek
    is_weekend = 1 if dow in [5, 6] else 0
    
    # Cyclical encodings
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    # Location-based features (normalized for Vijayawada region)
    lat_norm = (lat - 16.5) / 2.0
    lon_norm = (lon - 80.5) / 2.0
    
    # Create feature vector (22 features to match scaler)
    features = np.array([
        lat_norm, lon_norm,
        0, 0,  # Vehicle_Count, Traffic_Speed_kmh
        0, 0, 0,  # Accident_Report, Sentiment_Score, Ride_Sharing_Demand
        0, 0, 0,  # Parking_Availability, Emission_Levels_g_km, Energy_Consumption_L_h
        hour, dow, is_weekend,
        hour_sin, hour_cos,
        0, 0,  # Traffic_Light_State_Red, Traffic_Light_State_Yellow
        0, 0, 0,  # Weather_Condition_Fog, Weather_Condition_Rain, Weather_Condition_Snow
        0, 0  # Traffic_Condition_Low, Traffic_Condition_Medium
    ])
    
    return features

def predict_with_model(model_name, lat, lon, timestamp):
    """Make prediction using specified model"""
    try:
        if model_name not in models:
            return {'error': f'Model {model_name} not available'}
        
        # Preprocess input
        features = preprocess_location_data(lat, lon, timestamp)
        
        # Create sequence
        sequence_length = model_metadata.get('sequence_length', 24)
        sequence = np.tile(features, (sequence_length, 1))
        
        # Scale
        sequence_scaled = feature_scaler.transform(sequence)
        sequence_scaled = sequence_scaled[:, :18]  # Match model input
        sequence_scaled = sequence_scaled.reshape(1, sequence_length, -1)
        
        # Predict
        pred_scaled = models[model_name].predict(sequence_scaled, verbose=0)
        pred_original = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        
        # Apply realistic adjustments
        pred_original = apply_realistic_adjustments(pred_original, lat, lon, timestamp)
        
        return {
            'model': model_name,
            'prediction': float(pred_original),
            'timestamp': timestamp,
            'location': {'lat': lat, 'lon': lon}
        }
    except Exception as e:
        return {'error': str(e)}

def apply_realistic_adjustments(prediction, lat, lon, timestamp):
    """Apply time and location-based adjustments"""
    dt = pd.to_datetime(timestamp)
    hour = dt.hour
    minute = dt.minute
    
    # Ensure bounds
    prediction = max(0, min(100, prediction))
    
    # Peak hours adjustment
    if (7 <= hour <= 9) or (17 <= hour <= 19):
        minute_factor = 1.0 + (30 - abs(30 - minute)) / 150
        prediction = min(100, prediction * 1.3 * minute_factor + 10)
    elif 10 <= hour <= 16:
        minute_factor = 1.0 + (minute / 300)
        prediction = min(100, prediction * 1.15 * minute_factor + 5)
    elif 0 <= hour <= 5:
        prediction = max(0, prediction * 0.3)
    else:
        minute_factor = 1.0 + (minute / 600)
        prediction = min(100, prediction * 1.05 * minute_factor)
    
    # Location-based adjustment
    distance_from_center = np.sqrt((lat - 16.5)**2 + (lon - 80.6)**2)
    if distance_from_center < 0.05:
        prediction = min(100, prediction * 1.15 + 8)
    elif distance_from_center > 0.2:
        prediction = max(0, prediction * 0.7 - 5)
    
    # Add variation based on coordinates
    variation_seed = int((lat * 1000 + lon * 1000) % 100)
    if variation_seed % 3 == 0:
        prediction = max(0, prediction * 0.85)
    elif variation_seed % 3 == 1:
        prediction = min(100, prediction * 1.1)
    
    return max(0, min(100, prediction))

def create_ensemble_prediction(lat, lon, timestamp):
    """Create ensemble prediction from all available models"""
    predictions = []
    weights = {'lstm': 0.4, 'cnn_lstm': 0.6, 'best': 1.0}
    
    total_weight = 0
    weighted_sum = 0
    
    for model_name in models.keys():
        result = predict_with_model(model_name, lat, lon, timestamp)
        if 'error' not in result:
            weight = weights.get(model_name, 0.5)
            weighted_sum += result['prediction'] * weight
            total_weight += weight
            predictions.append(result)
    
    if total_weight > 0:
        ensemble_pred = weighted_sum / total_weight
        return {
            'model': 'ensemble',
            'prediction': float(ensemble_pred),
            'individual_predictions': predictions,
            'timestamp': timestamp,
            'location': {'lat': lat, 'lon': lon}
        }
    else:
        return {'error': 'No valid predictions available'}

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for traffic prediction"""
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['latitude', 'longitude', 'timestamp']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        lat = float(data['latitude'])
        lon = float(data['longitude'])
        timestamp = data['timestamp']
        model_name = data.get('model', 'ensemble')
        
        # Validate coordinates
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({'error': 'Invalid coordinates'}), 400
        
        # Make prediction
        if model_name == 'ensemble':
            result = create_ensemble_prediction(lat, lon, timestamp)
        else:
            result = predict_with_model(model_name, lat, lon, timestamp)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict_route', methods=['POST'])
def predict_route():
    """API endpoint for route-based traffic prediction"""
    try:
        data = request.get_json()
        
        if 'waypoints' not in data:
            return jsonify({'error': 'Missing waypoints'}), 400
        
        waypoints = data['waypoints']
        model_name = data.get('model', 'ensemble')
        
        if len(waypoints) < 2:
            return jsonify({'error': 'At least 2 waypoints required'}), 400
        
        route_predictions = []
        base_time = datetime.now()
        
        for i, waypoint in enumerate(waypoints):
            if 'latitude' not in waypoint or 'longitude' not in waypoint:
                return jsonify({'error': f'Invalid waypoint {i}'}), 400
            
            waypoint_time = base_time + timedelta(minutes=i * 5)
            
            if model_name == 'ensemble':
                result = create_ensemble_prediction(
                    waypoint['latitude'],
                    waypoint['longitude'],
                    waypoint_time.isoformat()
                )
            else:
                result = predict_with_model(
                    model_name,
                    waypoint['latitude'],
                    waypoint['longitude'],
                    waypoint_time.isoformat()
                )
            
            if 'error' not in result:
                route_predictions.append({
                    'waypoint': i,
                    'location': waypoint,
                    'prediction': result['prediction'],
                    'timestamp': waypoint_time.isoformat()
                })
        
        # Calculate route summary
        if route_predictions:
            avg_traffic = np.mean([p['prediction'] for p in route_predictions])
            max_traffic = max([p['prediction'] for p in route_predictions])
            min_traffic = min([p['prediction'] for p in route_predictions])
            
            return jsonify({
                'route_predictions': route_predictions,
                'summary': {
                    'average_traffic': float(avg_traffic),
                    'max_traffic': float(max_traffic),
                    'min_traffic': float(min_traffic),
                    'total_waypoints': len(route_predictions)
                }
            })
        else:
            return jsonify({'error': 'No valid predictions'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'available_models': list(models.keys()),
        'metadata': model_metadata,
        'total_models': len(models)
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'available_models': list(models.keys()),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    if load_all_models():
        print("🚀 Starting Enhanced Traffic Prediction API...")
        print("🌐 API available at: http://localhost:5001")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("❌ Failed to load models.")
