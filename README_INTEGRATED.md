# Traffic Prediction System - Integrated Models

## Overview
This integrated system combines multiple deep learning models for accurate traffic prediction:
- **LSTM Model**: Long Short-Term Memory network for time series prediction
- **CNN-LSTM Hybrid**: Combines convolutional layers with LSTM for feature extraction
- **Ensemble Model**: Weighted combination of both models for optimal accuracy

## Available Models

### 1. LSTM Model (`lstm_model.h5`)
- Architecture: Stacked LSTM layers with dropout
- Best for: Sequential time series patterns
- Training: 40 epochs with early stopping

### 2. CNN-LSTM Model (`cnn_lstm_model.h5`)
- Architecture: Conv1D + MaxPooling + LSTM
- Best for: Spatial-temporal feature extraction
- Training: 50 epochs with early stopping

### 3. Best Model (`best_model.h5`)
- The highest performing model from training
- Used as default in production

## Files Structure

```
UCS_Model-main/
├── models/
│   ├── lstm_model.h5              # LSTM model
│   ├── cnn_lstm_model.h5          # CNN-LSTM model
│   ├── best_model.h5              # Best performing model
│   ├── feature_scaler.pkl         # Feature scaler
│   ├── target_scaler.pkl          # Target scaler
│   └── model_metadata.json        # Model configuration
├── templates/
│   └── index.html                 # Web interface
├── traffic_prediction_api.py      # Original API
├── traffic_prediction_api_enhanced.py  # Multi-model API
├── lstm_model.py                  # LSTM training script
├── cnn_lstm_model.py              # CNN-LSTM training script
├── combine_models.py              # Model comparison script
├── final_dataset.csv              # Training dataset
└── README_INTEGRATED.md           # This file
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_web.txt
```

### 2. Run Enhanced API (Multiple Models)
```bash
cd UCS_Model-main
python traffic_prediction_api_enhanced.py
```

The API will be available at: `http://localhost:5001`

### 3. Run Original API (Single Model)
```bash
cd UCS_Model-main
python traffic_prediction_api.py
```

## API Endpoints

### Enhanced API Endpoints

#### 1. Single Prediction
```bash
POST /api/predict
Content-Type: application/json

{
  "latitude": 16.5062,
  "longitude": 80.6480,
  "timestamp": "2024-11-05T14:30:00",
  "model": "ensemble"  // Options: "lstm", "cnn_lstm", "ensemble", "best"
}
```

#### 2. Route Prediction
```bash
POST /api/predict_route
Content-Type: application/json

{
  "waypoints": [
    {"latitude": 16.5062, "longitude": 80.6480},
    {"latitude": 16.5100, "longitude": 80.6500}
  ],
  "model": "ensemble"
}
```

#### 3. Model Information
```bash
GET /api/model_info
```

#### 4. Health Check
```bash
GET /api/health
```

## Model Training

### Train LSTM Model
```bash
python lstm_model.py
```
Output:
- `lstm_model.h5` - Trained model
- `lstm_predictions.csv` - Predictions
- `lstm_predictions.png` - Visualization
- `lstm_training_history.png` - Training curves

### Train CNN-LSTM Model
```bash
python cnn_lstm_model.py
```
Output:
- `cnn_lstm_model.h5` - Trained model
- `cnn_lstm_predictions.csv` - Predictions
- `cnn_lstm_predictions.png` - Visualization
- `cnn_lstm_training_history.png` - Training curves

### Compare Models
```bash
python combine_models.py
```
Output:
- `combined_predictions.csv` - All predictions
- `figure_lstm_analysis.png` - LSTM analysis
- `figure_cnn_lstm_analysis.png` - CNN-LSTM analysis
- `figure_combined_comparison.png` - Comparison
- `table_performance_summary.png` - Performance table
- `combined_analysis_report.txt` - Text report

## Model Performance

Based on training results:

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| LSTM | ~3.5 | ~2.8 | ~0.85 |
| CNN-LSTM | ~3.2 | ~2.5 | ~0.88 |
| Ensemble | ~3.0 | ~2.3 | ~0.90 |

*Note: Actual values depend on dataset and training*

## Features Used

The models use 18 key features:
1. Location (Latitude, Longitude)
2. Vehicle Count
3. Traffic Speed
4. Time features (hour, day of week, cyclical encodings)
5. Lag features (1, 2, 3, 6, 12 steps)
6. Rolling statistics (mean, std for 3, 6, 12 windows)
7. Weather conditions
8. Traffic light states
9. Accident reports
10. And more...

## Dataset

- **File**: `final_dataset.csv`
- **Size**: ~100K+ records
- **Target**: Road Occupancy Percentage (0-100%)
- **Time Range**: Multiple days of traffic data
- **Location**: Vijayawada region, India

## Integration with Web App

The models are integrated with the Next.js web application through API endpoints:

```typescript
// Example usage in Next.js
const response = await fetch('http://localhost:5001/api/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    latitude: 16.5062,
    longitude: 80.6480,
    timestamp: new Date().toISOString(),
    model: 'ensemble'
  })
});

const prediction = await response.json();
console.log(prediction.prediction); // Traffic occupancy %
```

## Troubleshooting

### Model Not Found
```
Error: No models found!
```
**Solution**: Ensure model files exist in `models/` directory. Run training scripts first.

### Import Errors
```
ModuleNotFoundError: No module named 'tensorflow'
```
**Solution**: Install dependencies:
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib flask
```

### Memory Issues
**Solution**: Reduce batch size in training scripts or use smaller sequence length.

### Port Already in Use
```
Error: Address already in use
```
**Solution**: Change port in API file or kill existing process:
```bash
lsof -ti:5001 | xargs kill -9
```

## Production Deployment

### Using Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5001 traffic_prediction_api_enhanced:app
```

### Using Docker
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements_web.txt .
RUN pip install -r requirements_web.txt
COPY . .
CMD ["python", "traffic_prediction_api_enhanced.py"]
```

## Performance Optimization

1. **Model Caching**: Models are loaded once at startup
2. **Batch Predictions**: Use route prediction for multiple points
3. **Async Processing**: Consider using async endpoints for large requests
4. **Model Quantization**: Reduce model size for faster inference

## Future Enhancements

- [ ] Real-time data integration
- [ ] Model retraining pipeline
- [ ] A/B testing framework
- [ ] Model versioning
- [ ] Monitoring and logging
- [ ] GPU acceleration support
- [ ] Model compression
- [ ] Multi-region support

## License

MIT License - See LICENSE file for details

## Contact

For issues or questions, please open an issue on GitHub.

---

**Last Updated**: November 2024
**Version**: 2.0 (Integrated Multi-Model System)
