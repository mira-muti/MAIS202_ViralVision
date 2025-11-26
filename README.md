# 🎬 ViralVision - Video Virality Predictor

AI-powered video engagement prediction for Music and GRWM creators.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Installation

**Backend:**
```bash
pip install -r backend/requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

### Running

**Terminal 1 - Backend API:**
```bash
python backend/api_server.py
```
Server runs on: `http://localhost:8000`

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```
App runs on: `http://localhost:5173`

## 📁 Project Structure

```
MAIS202_ViralVision/
├── README.md
├── ARCHITECTURE.md
├── frontend/          # React + TypeScript frontend
├── backend/           # Python Flask API
│   ├── api_server.py
│   ├── predict.py
│   ├── extract_audio_features.py
│   ├── prediction_logger.py
│   └── requirements.txt
├── models/            # Trained ML models
│   ├── model.pkl
│   └── model_encoder.pkl
├── notebooks/         # Jupyter notebooks
└── data/              # Data files (gitignored)
    └── predictions_log.json
```

## 📡 API Endpoints

- `POST /api/predict` - Predict video engagement
- `GET /api/history` - Get prediction history
- `GET /api/health` - Health check

## 🎯 Features

- **Music Analysis**: Hook quality, audio energy, FFT features
- **GRWM Analysis**: Intro pacing, aesthetic consistency, motion
- **Real-time Predictions**: Upload video and get instant insights
- **Prediction History**: Track all predictions

## 📚 Documentation

See `ARCHITECTURE.md` for detailed architecture documentation.
