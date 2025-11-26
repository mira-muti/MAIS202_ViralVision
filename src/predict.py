"""
File: predict.py

Purpose:
    Inference pipeline for predicting video engagement from features.
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from src.extract_audio_features import extract_fft_features


MODEL_PATH = "models/model.pkl"
ENCODER_PATH = "models/model_encoder.pkl"

_model = None
_encoder = None


def _load_artifacts():
    """Load model and encoder from disk."""
    global _model, _encoder
    
    if _model is None or _encoder is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        if not os.path.exists(ENCODER_PATH):
            raise FileNotFoundError(f"Encoder not found at {ENCODER_PATH}")
        
        _model = joblib.load(MODEL_PATH)
        _encoder = joblib.load(ENCODER_PATH)


def predict_video(video_path, title, hashtags, niche):
    """
    Predict engagement level for a video.

    Args:
        video_path: Path to uploaded video
        title: String caption
        hashtags: String of hashtags
        niche: String category (music, food, summervibes, etc.)

    Returns:
        dict with:
            - label: "High" or "Low"
            - score: probability (0-100)
            - top_features: list of (feature, importance)
    """
    _load_artifacts()
    
    # Extract audio features
    fft_max_freq, fft_max_amp = extract_fft_features(video_path, seconds=5)
    
    # Compute text features
    caption_length = len(title) if title else 0
    hashtag_count = hashtags.count('#') if hashtags else 0
    
    # Build feature dataframe
    feature_data = {
        'engagement_ratio': [0.0],
        'caption_length': [caption_length],
        'hashtag_count': [hashtag_count],
        'fft_max_freq': [fft_max_freq],
        'fft_max_amp': [fft_max_amp],
        'niche': [niche]
    }
    df = pd.DataFrame(feature_data)
    
    # Apply one-hot encoding to niche
    niche_encoded = _encoder.transform(df[['niche']])
    niche_feature_names = _encoder.get_feature_names_out(['niche'])
    niche_df = pd.DataFrame(niche_encoded, columns=niche_feature_names, index=df.index)
    
    # Combine numeric features with encoded niche
    numeric_features = df[['engagement_ratio', 'caption_length', 'hashtag_count',
                          'fft_max_freq', 'fft_max_amp']]
    X = pd.concat([numeric_features, niche_df], axis=1)
    
    # Align columns to match training order (if model has feature_names_in_)
    if hasattr(_model, 'feature_names_in_'):
        X = X.reindex(columns=_model.feature_names_in_, fill_value=0)
    
    # Predict probability
    prob = _model.predict_proba(X)[0]
    prob_high = prob[1] if len(prob) > 1 else prob[0]
    
    # Convert to label
    label = "High" if prob_high >= 0.5 else "Low"
    score = float(prob_high * 100)
    
    # Extract feature importances
    if hasattr(_model, 'feature_importances_'):
        importances = _model.feature_importances_
        feature_names = X.columns.tolist()
        feature_importance_pairs = list(zip(feature_names, importances))
        top_features = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)
    else:
        top_features = []
    
    return {
        'label': label,
        'score': score,
        'top_features': top_features
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print("Usage:")
        print("  python3 src/predict.py <video_path> <title> <hashtags> <niche>")
        sys.exit(1)

    video_path = sys.argv[1]
    title = sys.argv[2]
    hashtags = sys.argv[3]
    niche = sys.argv[4]

    result = predict_video(video_path, title, hashtags, niche)
    print(result)
