#!/usr/bin/env python3
"""
LSTM Model for Traffic Prediction
Runs independently and saves model + generates predictions
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def load_and_prepare_data(csv_file_path):
    """Load and prepare the dataset"""
    df = pd.read_csv(csv_file_path)
    print(f"[LSTM] Loaded dataset: {df.shape}")
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    # Time features
    df['hour'] = df['Timestamp'].dt.hour
    df['dayofweek'] = df['Timestamp'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
    
    # Fill NaNs
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numerical_cols] = df[numerical_cols].fillna(method='ffill').fillna(method='bfill')
    
    # Smooth target
    df['Road_Occupancy_Percent'] = df['Road_Occupancy_Percent'].rolling(window=3, min_periods=1).mean()
    
    # Lag features
    for lag in [1, 2, 3, 6, 12]:
        df[f'lag_{lag}'] = df['Road_Occupancy_Percent'].shift(lag)
    
    # Rolling features
    for w in [3, 6, 12]:
        df[f'roll_mean_{w}'] = df['Road_Occupancy_Percent'].rolling(window=w, min_periods=1).mean()
        df[f'roll_std_{w}'] = df['Road_Occupancy_Percent'].rolling(window=w, min_periods=1).std().fillna(0)
    
    df = df.dropna().reset_index(drop=True)
    print(f"[LSTM] After preprocessing: {df.shape}")
    
    features_to_use = [
        'Road_Occupancy_Percent', 'Vehicle_Count', 'Traffic_Speed_kmh',
        'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
        'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
        'roll_mean_3', 'roll_mean_6', 'roll_mean_12',
        'roll_std_3', 'roll_std_6', 'roll_std_12'
    ]
    
    return df, features_to_use

def prepare_sequences(df, features_to_use, sequence_length=24):
    """Prepare sequences for LSTM"""
    df_agg = df.groupby('Timestamp')[features_to_use].mean().reset_index()
    
    feature_scaler = MinMaxScaler()
    all_features_scaled = feature_scaler.fit_transform(df_agg[features_to_use].values)
    
    target_scaler = MinMaxScaler()
    target_scaler.fit(df_agg[['Road_Occupancy_Percent']])
    
    X, y = [], []
    for i in range(len(all_features_scaled) - sequence_length):
        X.append(all_features_scaled[i:(i + sequence_length)])
        y.append(all_features_scaled[i + sequence_length, 0])
    
    X, y = np.array(X), np.array(y)
    
    train_split = int(len(X) * 0.8)
    X_train, X_test = X[:train_split], X[train_split:]
    y_train, y_test = y[:train_split], y[train_split:]
    
    # Sample for faster training
    sample_size = len(X_train) // 4
    indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_train, y_train = X_train[indices], y_train[indices]
    
    print(f"[LSTM] Training data: {X_train.shape}")
    return X_train, X_test, y_train, y_test, target_scaler

def build_lstm_model(input_shape):
    """Build LSTM model"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    print("\n" + "="*60)
    print("LSTM MODEL - Traffic Prediction")
    print("="*60 + "\n")
    
    # Load data
    df, features = load_and_prepare_data('final_dataset.csv')
    
    # Prepare sequences
    X_train, X_test, y_train, y_test, scaler = prepare_sequences(df, features)
    
    # Build model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    print(f"[LSTM] Model built with {model.count_params()} parameters")
    
    # Train
    print("\n[LSTM] Training started...")
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    
    history = model.fit(
        X_train, y_train,
        epochs=40,
        batch_size=128,
        validation_split=0.1,
        callbacks=[es, lr],
        verbose=1
    )
    
    # Predict
    print("\n[LSTM] Generating predictions...")
    y_pred_scaled = model.predict(X_test).reshape(-1)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    
    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print("\n" + "="*60)
    print("LSTM MODEL RESULTS")
    print("="*60)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    print("="*60 + "\n")
    
    # Save model
    model.save('lstm_model.h5')
    print("[LSTM] Model saved as 'lstm_model.h5'")
    
    # Save predictions
    results_df = pd.DataFrame({
        'Actual': y_true.flatten(),
        'Predicted': y_pred.flatten()
    })
    results_df.to_csv('lstm_predictions.csv', index=False)
    print("[LSTM] Predictions saved as 'lstm_predictions.csv'")
    
    # Plot predictions
    plt.figure(figsize=(15, 5))
    plt.plot(y_true[:200], label='Actual', alpha=0.7, linewidth=2)
    plt.plot(y_pred[:200], label='LSTM Prediction', linestyle='--', linewidth=2)
    plt.title('LSTM Model: Predictions vs Actual', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps')
    plt.ylabel('Road Occupancy %')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lstm_predictions.png', dpi=300, bbox_inches='tight')
    print("[LSTM] Plot saved as 'lstm_predictions.png'")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Training History - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('LSTM Training History - MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lstm_training_history.png', dpi=300, bbox_inches='tight')
    print("[LSTM] Training history saved as 'lstm_training_history.png'")
    
    print("\n[LSTM] ✓ Complete!\n")

if __name__ == "__main__":
    main()
