"""
Prediction pipeline for ViralVision.

Predicts engagement probability for uploaded videos using
trained GRWM/OOTD models.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import joblib
import numpy as np
import pandas as pd

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from extract_features import extract_audio_features, extract_visual_features
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Model paths
project_root = backend_dir.parent
MODELS_DIR = project_root / "backend" / "models"
MODEL_PATH = MODELS_DIR / "grwm_model.pkl"
ENCODER_PATH = MODELS_DIR / "grwm_encoder.pkl"
SCALER_PATH = MODELS_DIR / "grwm_scaler.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "grwm_feature_names.json"

_model = None
_encoder = None
_scaler = None
_feature_names = None


def _load_artifacts():
    """Load model, encoder, scaler, and feature names (cached)."""
    global _model, _encoder, _scaler, _feature_names
    
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        if not ENCODER_PATH.exists():
            raise FileNotFoundError(f"Encoder not found at {ENCODER_PATH}")
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
        
        _model = joblib.load(MODEL_PATH)
        _encoder = joblib.load(ENCODER_PATH)
        _scaler = joblib.load(SCALER_PATH)
        
        # Load feature names
        if FEATURE_NAMES_PATH.exists():
            with open(FEATURE_NAMES_PATH, 'r') as f:
                _feature_names = json.load(f)
        else:
            _feature_names = None


def _build_feature_vector(
    audio_features: Dict[str, float],
    visual_features: Dict[str, float],
    caption_length: int,
    hashtag_count: int,
    niche: str,
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame matching training order.
    
    Args:
        audio_features: Dictionary of audio features
        visual_features: Dictionary of visual features
        caption_length: Caption length
        hashtag_count: Hashtag count
        niche: Niche label (GRWM, OOTD)
        
    Returns:
        DataFrame with features in correct order
    """
    # Build numeric features
    data = {
        "engagement_ratio": [0.0],  # Set to 0 at inference
        "caption_length": [caption_length],
        "hashtag_count": [hashtag_count],
        "avg_brightness": [visual_features.get("avg_brightness", 0.0)],
        "color_std_dev": [visual_features.get("color_std_dev", 0.0)],
        "motion_intensity": [visual_features.get("motion_intensity", 0.0)],
        "scene_change_rate": [visual_features.get("scene_change_rate", 0.0)],
        "hue_entropy": [visual_features.get("hue_entropy", 0.0)],
        "face_present": [visual_features.get("face_present", 0.0)],
        "text_overlay_present": [visual_features.get("text_overlay_present", 0.0)],
        "rms_energy": [audio_features.get("rms_energy", 0.0)],
        "zcr": [audio_features.get("zcr", 0.0)],
        "spectral_centroid": [audio_features.get("spectral_centroid", 0.0)],
        "spectral_rolloff": [audio_features.get("spectral_rolloff", 0.0)],
        "fft_max_freq": [audio_features.get("fft_max_freq", 0.0)],
        "fft_max_amp": [audio_features.get("fft_max_amp", 0.0)],
        "niche": [niche],
    }
    
    df = pd.DataFrame(data)
    
    # Encode niche
    niche_encoded = _encoder.transform(df[["niche"]])
    niche_feature_names = _encoder.get_feature_names_out(["niche"])
    niche_df = pd.DataFrame(niche_encoded, columns=niche_feature_names, index=df.index)
    
    # Combine numeric + encoded niche
    numeric_cols = [
        "engagement_ratio", "caption_length", "hashtag_count",
        "avg_brightness", "color_std_dev", "motion_intensity", "scene_change_rate",
        "hue_entropy", "face_present", "text_overlay_present",
        "rms_energy", "zcr", "spectral_centroid", "spectral_rolloff",
        "fft_max_freq", "fft_max_amp"
    ]
    numeric_df = df[numeric_cols]
    X = pd.concat([numeric_df, niche_df], axis=1)
    
    # Standardize numeric features
    X_scaled = X.copy()
    X_scaled[numeric_cols] = _scaler.transform(X[numeric_cols])
    
    # Align with model feature order if available
    if hasattr(_model, "feature_names_in_"):
        X_scaled = X_scaled.reindex(columns=_model.feature_names_in_, fill_value=0)
    elif _feature_names:
        X_scaled = X_scaled.reindex(columns=_feature_names, fill_value=0)
    
    return X_scaled


def _get_feature_importances(X: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract top positive and negative features from model.
    
    Returns:
        Tuple of (top_positive_features, top_negative_features)
    """
    top_positive = []
    top_negative = []
    
    if not hasattr(_model, "feature_importances_"):
        return top_positive, top_negative
    
    importances = _model.feature_importances_
    names = X.columns.tolist()
    pairs = list(zip(names, importances))
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    
    # Top 5 positive
    for name, imp in sorted_pairs[:5]:
        top_positive.append({"feature": name, "importance": float(imp)})
    
    # Bottom 5 negative (excluding niche features)
    if len(sorted_pairs) > 5:
        avg_imp = float(np.mean(importances))
        bottom = [
            (name, imp)
            for name, imp in sorted_pairs[-10:]
            if "niche" not in name.lower() and imp < avg_imp * 0.7
        ]
        bottom = bottom[:5]
        for name, imp in bottom:
            top_negative.append({
                "feature": name,
                "importance": float(max(0.0, avg_imp - imp))
            })
    
    return top_positive, top_negative


def _generate_recommendations(
    audio_features: Dict[str, float],
    visual_features: Dict[str, float],
    caption_length: int,
    hashtag_count: int,
    niche: str,
) -> List[str]:
    """
    Generate human-readable recommendations based on features.
    
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # Caption recommendations
    if caption_length < 20:
        recommendations.append(
            f"Try adding more context to your caption – {niche} viewers appreciate details."
        )
    elif caption_length > 120:
        recommendations.append(
            "Caption is long – consider making it more concise while keeping key information."
        )
    
    # Hashtag recommendations
    if hashtag_count == 0:
        recommendations.append(
            f"Add 2–4 relevant hashtags to reach more {niche} viewers."
        )
    elif hashtag_count > 12:
        recommendations.append(
            "You have many hashtags – focus on 3–6 highly relevant ones."
        )
    
    # Audio recommendations
    rms_energy = audio_features.get("rms_energy", 0.0)
    if rms_energy < 0.01:
        recommendations.append(
            "Audio energy is very low – consider improving audio quality or adding background music."
        )
    
    # Visual recommendations
    motion_intensity = visual_features.get("motion_intensity", 0.0)
    if motion_intensity < 5.0:
        recommendations.append(
            "Video has low motion – consider adding more dynamic movement or cuts."
        )
    
    if not recommendations:
        recommendations.append("Overall features look solid – keep this style in future posts.")
    
    return recommendations


def predict_video(
    video_path: str,
    title: str,
    hashtags: str,
    niche: str = "GRWM",
) -> Dict:
    """
    Predict engagement for a video.
    
    Args:
        video_path: Path to video file
        title: Video title/caption
        hashtags: Hashtag string
        niche: Niche label (GRWM, OOTD)
        
    Returns:
        Dictionary with prediction results:
        - niche: Niche label
        - prediction: "High" or "Low"
        - probability: Probability of high engagement (0-1)
        - prob_high: Probability of high engagement
        - prob_low: Probability of low engagement
        - features: Raw feature dictionaries
        - top_positive_features: List of top positive features
        - top_negative_features: List of top negative features
        - recommendations: List of improvement suggestions
        - notes: Human-readable explanation
    """
    # Load artifacts
    _load_artifacts()
    
    niche_upper = (niche or "GRWM").upper()
    
    # Extract features (first 3 seconds)
    audio_features = extract_audio_features(video_path, max_seconds=3.0)
    visual_features = extract_visual_features(video_path, max_seconds=3.0)
    
    # Compute text features
    caption_length = len(title.strip()) if title else 0
    hashtag_count = 0
    if hashtags:
        hashtag_tokens = hashtags.split()
        hashtag_count = sum(1 for token in hashtag_tokens if token.startswith("#"))
    
    # Build feature vector
    X = _build_feature_vector(
        audio_features, visual_features, caption_length, hashtag_count, niche_upper
    )
    
    # Predict
    prob = _model.predict_proba(X)[0]
    prob_low = float(prob[0])
    prob_high = float(prob[1]) if len(prob) > 1 else float(prob[0])
    label = "High" if prob_high >= 0.5 else "Low"
    
    # Get feature importances
    top_positive, top_negative = _get_feature_importances(X)
    
    # Generate recommendations
    recommendations = _generate_recommendations(
        audio_features, visual_features, caption_length, hashtag_count, niche_upper
    )
    
    # Build result
    result = {
        "niche": niche_upper,
        "prediction": label,
        "probability": prob_high,
        "prob_high": prob_high,
        "prob_low": prob_low,
        "features": {
            "audio": audio_features,
            "visual": visual_features,
            "text": {
                "caption_length": caption_length,
                "hashtag_count": hashtag_count,
            }
        },
        "top_positive_features": top_positive,
        "top_negative_features": top_negative,
        "recommendations": recommendations,
        "notes": f"Predicted {label} engagement ({prob_high*100:.1f}% probability) using {niche_upper} model.",
    }
    
    return result


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage:")
        print("  python3 backend/predict.py <video_path> <title> <hashtags> <niche>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    title = sys.argv[2]
    hashtags = sys.argv[3]
    niche = sys.argv[4]
    
    res = predict_video(video_path, title, hashtags, niche)
    print(json.dumps(res, indent=2))
