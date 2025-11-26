"""
GRWM_ENGINE - ML-based engagement prediction for GRWM-style content.

This engine uses the trained RandomForest model and encoder to predict
High vs Low engagement for non-music niches.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from extract_audio_features import extract_audio_feature_dict
from extract_visual_features import extract_visual_features

project_root = Path(__file__).parent.parent
MODELS_DIR = project_root / "backend" / "models"
MODEL_PATH = MODELS_DIR / "grwm_model.pkl"
ENCODER_PATH = MODELS_DIR / "grwm_encoder.pkl"

_model = None
_encoder = None


def _load_artifacts():
    global _model, _encoder
    if _model is None or _encoder is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"GRWM model not found at {MODEL_PATH}")
        if not ENCODER_PATH.exists():
            raise FileNotFoundError(f"GRWM encoder not found at {ENCODER_PATH}")
        _model = joblib.load(MODEL_PATH)
        _encoder = joblib.load(ENCODER_PATH)


def _build_feature_vector(
    audio_features: Dict[str, float],
    caption_length: int,
    hashtag_count: int,
    niche: str,
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame matching training order.
    """
    fft_max_freq = float(audio_features.get("fft_max_freq", 0.0) or 0.0)
    fft_max_amp = float(audio_features.get("fft_max_amp", 0.0) or 0.0)

    data = {
        "engagement_ratio": [0.0],
        "caption_length": [caption_length],
        "hashtag_count": [hashtag_count],
        "fft_max_freq": [fft_max_freq],
        "fft_max_amp": [fft_max_amp],
        "niche": [niche],
    }
    df = pd.DataFrame(data)

    niche_encoded = _encoder.transform(df[["niche"]])
    niche_feature_names = _encoder.get_feature_names_out(["niche"])
    niche_df = pd.DataFrame(niche_encoded, columns=niche_feature_names, index=df.index)

    numeric = df[
        ["engagement_ratio", "caption_length", "hashtag_count", "fft_max_freq", "fft_max_amp"]
    ]
    X = pd.concat([numeric, niche_df], axis=1)

    if hasattr(_model, "feature_names_in_"):
        X = X.reindex(columns=_model.feature_names_in_, fill_value=0)

    return X


def _feature_importances(X: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    """
    Derive top positive and negative features based on RandomForest importances.
    """
    top_positive: List[Dict] = []
    top_negative: List[Dict] = []

    if not hasattr(_model, "feature_importances_"):
        return top_positive, top_negative

    importances = _model.feature_importances_
    names = X.columns.tolist()
    pairs = list(zip(names, importances))
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

    # Top 5 strongest features
    for name, imp in sorted_pairs[:5]:
        top_positive.append({"feature": name, "importance": float(imp)})

    # Bottom 5 weakest (excluding niche features)
    if len(sorted_pairs) > 5:
        avg_imp = float(np.mean(importances))
        bottom = [
            (name, imp)
            for name, imp in sorted_pairs[-10:]
            if "niche" not in name.lower() and imp < avg_imp * 0.7
        ]
        bottom = bottom[:5]
        for name, imp in bottom:
            top_negative.append(
                {"feature": name, "importance": float(max(0.0, avg_imp - imp))}
            )

    return top_positive, top_negative


def predict_grwm_video(
    video_path: str,
    title: str,
    hashtags: str,
    niche: str,
) -> Dict:
    """
    Predict engagement for GRWM-style content using the ML model.
    """
    _load_artifacts()

    # Audio + visual features
    audio_features = extract_audio_feature_dict(video_path, seconds=5)
    visual_features = extract_visual_features(video_path, max_seconds=5, frame_sample_rate=10)

    caption_length = len(title.strip()) if title else 0
    hashtag_count = 0
    if hashtags:
        hashtag_tokens = hashtags.split()
        hashtag_count = sum(1 for token in hashtag_tokens if token.startswith("#"))

    X = _build_feature_vector(audio_features, caption_length, hashtag_count, niche)

    prob = _model.predict_proba(X)[0]
    prob_low = float(prob[0])
    prob_high = float(prob[1]) if len(prob) > 1 else float(prob[0])
    label = "High" if prob_high >= 0.5 else "Low"

    top_pos, top_neg = _feature_importances(X)

    # Simple recommendations based on text features
    recommendations: List[str] = []
    if caption_length < 20:
        recommendations.append(
            "Try adding more context or storytelling to your caption – GRWM viewers love details."
        )
    elif caption_length > 120:
        recommendations.append(
            "Caption is long – consider making it more concise while keeping the key story."
        )

    if hashtag_count == 0:
        recommendations.append(
            "Add 2–4 lifestyle/aesthetic hashtags to reach more viewers."
        )
    elif hashtag_count > 12:
        recommendations.append(
            "You have many hashtags – focus on 3–6 highly relevant ones for your GRWM style."
        )

    if not recommendations:
        recommendations.append("Overall metadata looks solid – keep this style in future posts.")

    score = prob_high * 100.0

    result = {
        "engine": "grwm",
        "label": label,
        "probability": prob_high,
        "score": score,
        "features_positive": top_pos,
        "features_negative": top_neg,
        "raw_features": {
            "audio": audio_features,
            "visual": visual_features,
            "text": {
                "caption_length": caption_length,
                "hashtag_count": hashtag_count,
                "niche": niche,
            },
        },
        # Legacy fields for frontend compatibility
        "prob_high": prob_high,
        "prob_low": prob_low,
        "top_positive_features": top_pos,
        "top_negative_features": top_neg,
        "recommendations": recommendations,
    }

    return result


