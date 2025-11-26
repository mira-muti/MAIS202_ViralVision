"""
File: predict.py

Purpose:
    Inference pipeline for predicting video engagement from features.
    Now includes audio and visual feature extraction with explainability.
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from typing import Dict, List
from extract_audio_features import extract_audio_feature_dict, extract_fft_features
from extract_visual_features import extract_visual_features
from prediction_logger import log_prediction

# Get project root (one level up from backend/)
project_root = Path(__file__).parent.parent
MODEL_PATH = str(project_root / "models" / "model.pkl")
ENCODER_PATH = str(project_root / "models" / "model_encoder.pkl")

_model = None
_encoder = None


def get_model_and_encoder():
    """
    Load model and encoder from disk with caching.
    
    Caches loaded artifacts in module-level globals to avoid reloading
    on repeated calls.
    
    Raises:
        FileNotFoundError: If model or encoder files don't exist
    """
    global _model, _encoder
    
    if _model is None or _encoder is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        if not os.path.exists(ENCODER_PATH):
            raise FileNotFoundError(f"Encoder not found at {ENCODER_PATH}")
        
        _model = joblib.load(MODEL_PATH)
        _encoder = joblib.load(ENCODER_PATH)


def _generate_explanations(
    audio_features: Dict[str, float],
    visual_features: Dict[str, float],
    niche: str
) -> tuple[List[str], List[str]]:
    """
    Generate human-readable explanations based on audio and visual features.
    
    Args:
        audio_features: Dictionary from extract_audio_feature_dict
        visual_features: Dictionary from extract_visual_features
        niche: Content niche ("music" or "GRWM")
        
    Returns:
        Tuple of (positives, improvements) - lists of explanation strings
    """
    positives = []
    improvements = []
    is_music = niche.lower() == "music"
    
    # Audio feature explanations
    rms_energy = audio_features.get("rms_energy", 0.0)
    spectral_centroid = audio_features.get("spectral_centroid", 0.0)
    fft_max_amp = audio_features.get("fft_max_amp", 0.0)
    tempo = audio_features.get("tempo", 0.0)
    
    # RMS Energy (loudness/clarity)
    if rms_energy > 0.1:
        positives.append("Audio is loud and clear – great for grabbing attention 🎧")
    elif rms_energy < 0.05:
        improvements.append("Audio could be louder. Consider increasing volume or using clearer audio sources.")
    
    # Spectral Centroid (brightness)
    if spectral_centroid > 2000:
        positives.append("Spectral centroid suggests a bright, modern sound signature ✨")
    elif spectral_centroid < 1000:
        improvements.append("Audio sounds muted. Consider using brighter, more energetic tracks.")
    
    # FFT Amplitude (energy)
    if fft_max_amp > 100000000:
        positives.append("Strong audio energy detected. This contributes positively 🔥")
    elif fft_max_amp < 10000000:
        improvements.append("Audio energy is low. Consider using tracks with more presence.")
    
    # Tempo
    if tempo > 0:
        if 120 <= tempo <= 140:
            positives.append("Tempo is in the sweet spot for engaging content (120-140 BPM) 🎵")
        elif tempo < 80:
            improvements.append("Tempo is quite slow. Consider using more upbeat tracks for better engagement.")
        elif tempo > 160:
            improvements.append("Tempo is very fast. Consider slightly slower tracks for better retention.")
    
    # Visual feature explanations
    avg_brightness = visual_features.get("avg_brightness", 0.0)
    motion_intensity = visual_features.get("motion_intensity", 0.0)
    color_variance = visual_features.get("avg_color_variance", 0.0)
    
    # Brightness
    if avg_brightness > 120:
        positives.append("Brightness is excellent – well-lit content performs better 💡")
    elif avg_brightness < 80:
        if is_music:
            improvements.append("Brightness is below typical music content – consider better lighting.")
        else:
            improvements.append("Brightness is below typical GRWM content – consider better lighting.")
    elif avg_brightness < 100:
        improvements.append("Lighting could be improved. Brighter videos tend to perform better.")
    
    # Motion intensity
    if motion_intensity > 5.0:
        positives.append("Good motion intensity – dynamic content keeps viewers engaged 🎬")
    elif motion_intensity < 1.0:
        if is_music:
            improvements.append("Motion is very low – consider adding more dynamic camera work or cuts.")
        else:
            improvements.append("Motion is very low – this might feel static for GRWM videos. Add more pacing changes.")
    elif motion_intensity < 2.0:
        improvements.append("Motion could be increased. More dynamic content tends to perform better.")
    
    # Color variance
    if color_variance > 5000:
        positives.append("Good color diversity – visually engaging content 🎨")
    elif color_variance < 1000:
        improvements.append("Color palette is quite uniform. Consider adding more visual variety.")
    
    # Niche-specific explanations
    fft_max_freq = audio_features.get("fft_max_freq", 0.0)
    if is_music:
        if fft_max_freq > 500:
            positives.append("Audio frequency matches trending high-energy tracks 🎸")
        elif fft_max_freq < 100:
            improvements.append("Consider using more energetic audio to match trending content patterns.")
    else:  # GRWM
        if avg_brightness > 100 and motion_intensity > 2.0:
            positives.append("Aesthetic consistency and pacing are on point 💅")
        if avg_brightness < 90:
            improvements.append("Lighting could be brighter. Aesthetic consistency matters for GRWM content.")
    
    # Ensure we have at least some feedback
    if not positives:
        positives.append("Video has potential – keep creating! 🎥")
    if not improvements:
        improvements.append("Continue refining your content for even better results.")
    
    return positives[:5], improvements[:5]  # Limit to top 5 each


def predict_video(video_path: str, title: str, hashtags: str, niche: str, video_filename: str = None) -> Dict:
    """
    Predict engagement level for a video with comprehensive feature extraction.
    
    This function:
    1. Extracts audio features (FFT + extended features)
    2. Extracts visual features (brightness, motion, color)
    3. Builds feature vector matching training data
    4. Runs model prediction
    5. Generates explanations based on all features
    
    Args:
        video_path: Path to uploaded video file
        title: String caption/title
        hashtags: String of hashtags (e.g., "#music #cover")
        niche: Content niche ("music" or "GRWM")
        video_filename: Optional filename for logging
        
    Returns:
        Dictionary with:
            - label: "High" or "Low"
            - prob_high: probability of high engagement (0-1)
            - prob_low: probability of low engagement (0-1)
            - audio_features: dict from extract_audio_feature_dict
            - visual_features: dict from extract_visual_features
            - text_features: dict with caption_length, hashtag_count, niche
            - positives: list of positive feedback strings
            - improvements: list of improvement suggestion strings
    """
    get_model_and_encoder()
    
    # 1) Extract audio features (first 5 seconds)
    try:
        audio_features = extract_audio_feature_dict(video_path, seconds=5)
    except Exception as e:
        raise RuntimeError(f"Error extracting audio features: {str(e)}")
    
    # 2) Extract visual features (first 5 seconds)
    try:
        visual_features = extract_visual_features(video_path, max_seconds=5, frame_sample_rate=10)
    except Exception as e:
        raise RuntimeError(f"Error extracting visual features: {str(e)}")
    
    # 3) Compute text features
    caption_length = len(title.strip()) if title else 0
    
    # Count hashtags (tokens starting with "#")
    hashtag_count = 0
    if hashtags:
        hashtag_tokens = hashtags.split()
        hashtag_count = sum(1 for token in hashtag_tokens if token.startswith("#"))
    
    # 4) Build feature vector for model (matching training structure)
    # Model expects: [engagement_ratio, caption_length, hashtag_count, fft_max_freq, fft_max_amp, niche_encoded...]
    # At inference, engagement_ratio is set to 0 (placeholder)
    fft_max_freq = audio_features["fft_max_freq"]
    fft_max_amp = audio_features["fft_max_amp"]
    
    feature_data = {
        'engagement_ratio': [0.0],  # Placeholder, not used at inference
        'caption_length': [caption_length],
        'hashtag_count': [hashtag_count],
        'fft_max_freq': [fft_max_freq],
        'fft_max_amp': [fft_max_amp],
        'niche': [niche]
    }
    df = pd.DataFrame(feature_data)
    
    # Apply one-hot encoding to niche (same as training)
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
    
    # 5) Predict probability
    prob = _model.predict_proba(X)[0]
    # prob[0] = Low, prob[1] = High (based on label encoding: High=1, Low=0)
    prob_low = float(prob[0])
    prob_high = float(prob[1]) if len(prob) > 1 else float(prob[0])
    
    # Convert to label
    label = "High" if prob_high >= 0.5 else "Low"
    
    # 6) Generate explanations
    positives, improvements = _generate_explanations(audio_features, visual_features, niche)
    
    # Build result dictionary
    result = {
        'label': label,
        'prob_high': prob_high,
        'prob_low': prob_low,
        'audio_features': audio_features,
        'visual_features': visual_features,
        'text_features': {
            'caption_length': caption_length,
            'hashtag_count': hashtag_count,
            'niche': niche
        },
        'positives': positives,
        'improvements': improvements
    }
    
    # Log the prediction (using old format for compatibility)
    if video_filename:
        log_result = {
            'label': label,
            'prob_high': prob_high,
            'prob_low': prob_low
        }
        log_prediction(video_filename, title, niche, log_result)
    
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print("Usage:")
        print("  python3 backend/predict.py <video_path> <title> <hashtags> <niche>")
        sys.exit(1)

    video_path = sys.argv[1]
    title = sys.argv[2]
    hashtags = sys.argv[3]
    niche = sys.argv[4]

    result = predict_video(video_path, title, hashtags, niche)
    
    # Pretty print results
    import json
    print(json.dumps(result, indent=2))
