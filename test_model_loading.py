#!/usr/bin/env python3
"""
Quick test script to verify model loading
Run this before starting the Flask API
"""

import os
import sys
import tensorflow as tf
import joblib
import json

print("=" * 60)
print("   MODEL LOADING TEST")
print("=" * 60)
print()

# Check current directory
current_dir = os.getcwd()
print(f"📁 Current directory: {current_dir}")
print()

# Check if we're in the right directory
if not os.path.exists('traffic_prediction_api.py'):
    print("❌ ERROR: Not in UCS_Model-main directory!")
    print(f"   Current: {current_dir}")
    print(f"   Expected to find: traffic_prediction_api.py")
    print()
    print("💡 Solution: cd to UCS_Model-main directory")
    sys.exit(1)

print("✅ In correct directory")
print()

# Check for model files
print("🔍 Checking for required files...")
required_files = [
    'models/traffic_prediction_model.h5',
    'models/feature_scaler.pkl',
    'models/target_scaler.pkl',
    'models/model_metadata.json'
]

missing_files = []
for file in required_files:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"   ✅ {file} ({size:,} bytes)")
    else:
        print(f"   ❌ {file} - NOT FOUND")
        missing_files.append(file)

print()

if missing_files:
    print("❌ ERROR: Missing model files!")
    print("   Missing files:")
    for file in missing_files:
        print(f"      - {file}")
    print()
    print("💡 Solution: Train the model using Capstone_LSTM_CNN-GRU_Notebook.ipynb")
    sys.exit(1)

# Try loading TensorFlow model
print("🧪 Testing TensorFlow model loading...")
try:
    # Load without compiling (fix for metric compatibility)
    model = tf.keras.models.load_model('models/traffic_prediction_model.h5', compile=False)
    print("   ✅ Model architecture loaded")
    
    # Compile with compatible metrics
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    print("   ✅ Model compiled successfully")
    
    # Print model summary
    print()
    print("📊 Model Summary:")
    model.summary()
    
except Exception as e:
    print(f"   ❌ Failed to load model: {e}")
    sys.exit(1)

print()

# Try loading scalers
print("🧪 Testing scaler loading...")
try:
    feature_scaler = joblib.load('models/feature_scaler.pkl')
    print("   ✅ Feature scaler loaded")
    
    target_scaler = joblib.load('models/target_scaler.pkl')
    print("   ✅ Target scaler loaded")
    
except Exception as e:
    print(f"   ❌ Failed to load scalers: {e}")
    sys.exit(1)

print()

# Try loading metadata
print("🧪 Testing metadata loading...")
try:
    with open('models/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print("   ✅ Metadata loaded")
    print(f"   📊 Model type: {metadata.get('model_type', 'Unknown')}")
    print(f"   📊 Sequence length: {metadata.get('sequence_length', 'Unknown')}")
    print(f"   📊 Features: {metadata.get('n_features', 'Unknown')}")
    
    if 'model_performance' in metadata:
        perf = metadata['model_performance']
        print(f"   📊 RMSE: {perf.get('rmse', 'N/A')}")
        print(f"   📊 MAE: {perf.get('mae', 'N/A')}")
        print(f"   📊 R²: {perf.get('r2', 'N/A')}")
    
except Exception as e:
    print(f"   ❌ Failed to load metadata: {e}")
    sys.exit(1)

print()
print("=" * 60)
print("   ✅ ALL TESTS PASSED!")
print("=" * 60)
print()
print("🚀 You can now start the Flask API:")
print("   python traffic_prediction_api.py")
print()
