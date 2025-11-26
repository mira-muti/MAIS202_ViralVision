# 🏗️ ViralVision Architecture

## Overview

ViralVision is a full-stack ML application with a React frontend and Python Flask backend.

## System Architecture

```
┌─────────────┐         HTTP/REST         ┌──────────────┐
│   React     │  ───────────────────────>  │   Flask API  │
│  Frontend   │  <───────────────────────  │   Server     │
│ (Port 5173) │         JSON Response      │ (Port 8000)  │
└─────────────┘                            └──────────────┘
                                                    │
                                                    ▼
                                            ┌──────────────┐
                                            │   ML Model   │
                                            │  (Random     │
                                            │   Forest)    │
                                            └──────────────┘
```

## Directory Structure

### `/frontend`
React + TypeScript application
- `src/pages/` - Page components (Landing, Analyze, Results, History)
- `src/components/` - Reusable UI components
- `src/api/` - API client functions

### `/backend`
Python Flask API server
- `api_server.py` - Main Flask application
- `predict.py` - Main prediction function
- `extract_audio_features.py` - Audio feature extraction
- `prediction_logger.py` - Prediction history logging
- `requirements.txt` - Python dependencies

### `/models`
Trained ML models
- `model.pkl` - Random Forest classifier
- `model_encoder.pkl` - OneHotEncoder for niches

### `/notebooks`
Jupyter notebooks for EDA and testing

### `/data`
Data storage (gitignored)
- `uploads/` - Temporary video uploads
- `predictions_log.json` - Prediction history

## Data Flow

1. **User uploads video** via React frontend
2. **Frontend sends POST** to `/api/predict` with video file and metadata
3. **Backend processes**:
   - Extracts audio features (FFT: max_freq, max_amp)
   - Computes text features (caption_length, hashtag_count)
   - Loads trained model
   - Runs prediction
4. **Backend returns** JSON with:
   - Prediction label (High/Low)
   - Probabilities
   - Feature importances
   - Recommendations
5. **Frontend displays** results with visualizations

## ML Pipeline

### Features Extracted

**Audio Features:**
- `fft_max_freq` - Dominant frequency (brightness)
- `fft_max_amp` - Maximum amplitude (energy)

**Text Features:**
- `caption_length` - Length of video title/caption
- `hashtag_count` - Number of hashtags
- `engagement_ratio` - Placeholder (0 at inference)

**Categorical:**
- `niche` - One-hot encoded (Music, GRWM)

### Model

- **Algorithm**: Random Forest Classifier
- **Input**: 5 numeric features + one-hot encoded niche
- **Output**: Binary classification (High/Low engagement)
- **Training**: Uses `data/final_dataset.csv`

## API Endpoints

### `POST /api/predict`

**Request:**
- `video`: File (multipart/form-data)
- `title`: String
- `hashtags`: String
- `niche`: String ("music" or "GRWM")

**Response:**
```json
{
  "label": "High",
  "prob_high": 0.87,
  "prob_low": 0.13,
  "top_positive_features": [...],
  "top_negative_features": [...],
  "recommendations": [...]
}
```

### `GET /api/history`

Returns array of past predictions.

### `GET /api/health`

Returns `{"status": "ok"}`

## Technology Stack

**Frontend:**
- React 18
- TypeScript
- TailwindCSS
- Vite
- Framer Motion

**Backend:**
- Python 3.8+
- Flask
- scikit-learn
- MoviePy
- pandas, numpy
