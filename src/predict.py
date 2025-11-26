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
from src.prediction_logger import log_prediction


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


def predict_video(video_path, title, hashtags, niche, video_filename=None):
    """
    Predict engagement level for a video.

    Args:
        video_path: Path to uploaded video
        title: String caption
        hashtags: String of hashtags
        niche: String category (music, food, summervibes, etc.)
        video_filename: Optional filename for logging

    Returns:
        dict with:
            - label: "High" or "Low"
            - prob_high: probability of high engagement (0-1)
            - prob_low: probability of low engagement (0-1)
            - top_positive_features: list of positive feature contributions
            - top_negative_features: list of negative feature contributions
            - recommendations: list of improvement suggestions
            - raw_feature_importances: dict of all feature importances
    """
    _load_artifacts()
    
    # Extract audio features (analyze first 60 seconds for better accuracy)
    fft_max_freq, fft_max_amp = extract_fft_features(video_path, seconds=60)
    
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
    # prob[0] = Low, prob[1] = High (based on label encoding: High=1, Low=0)
    prob_low = float(prob[0])
    prob_high = float(prob[1]) if len(prob) > 1 else float(prob[0])
    
    # Convert to label
    label = "High" if prob_high >= 0.5 else "Low"
    
    # Extract feature importances
    raw_feature_importances = {}
    top_positive_features = []
    top_negative_features = []
    
    if hasattr(_model, 'feature_importances_'):
        importances = _model.feature_importances_
        feature_names = X.columns.tolist()
        
        # Store raw importances
        for name, importance in zip(feature_names, importances):
            raw_feature_importances[name] = float(importance)
        
        # For RandomForest, all importances are positive
        # Top features = highest importance (helped performance)
        # Bottom features = lowest importance (less impactful, areas to improve)
        feature_importance_pairs = list(zip(feature_names, importances))
        sorted_features = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)
        
        # Top 5 as positive (helped)
        top_positive_features = [
            {'feature': name, 'importance': float(imp)}
            for name, imp in sorted_features[:5]
        ]
        
        # Bottom 5 as "areas to improve" (less impactful)
        # Only include if there are enough features and they're significantly lower
        # EXCLUDE niche features from negative list (they're not actionable improvements)
        if len(sorted_features) > 5:
            avg_importance = sum(imp for _, imp in sorted_features) / len(sorted_features)
            bottom_features = sorted_features[-5:]
            top_negative_features = [
                {'feature': name, 'importance': float(avg_importance - imp)}
                for name, imp in bottom_features
                if imp < avg_importance * 0.5 and 'niche' not in name.lower()  # Exclude niche features
            ][:5]
    
    # Generate recommendations based on features and niche
    recommendations = _generate_recommendations(
        caption_length, hashtag_count, fft_max_freq, fft_max_amp, 
        prob_high, top_positive_features, top_negative_features, niche
    )
    
    result = {
        'label': label,
        'prob_high': prob_high,
        'prob_low': prob_low,
        'top_positive_features': top_positive_features,
        'top_negative_features': top_negative_features,
        'recommendations': recommendations,
        'raw_feature_importances': raw_feature_importances
    }
    
    # Log the prediction
    if video_filename:
        log_prediction(video_filename, title, niche, result)
    
    return result


def _generate_recommendations(caption_length, hashtag_count, fft_max_freq, fft_max_amp, 
                              prob_high, positive_features, negative_features, niche='music'):
    """Generate personalized recommendations based on features and niche."""
    recommendations = []
    is_music = niche.lower() == 'music'
    
    if is_music:
        # Music-specific recommendations
        if caption_length < 20:
            recommendations.append("Try adding storytelling or context to your caption. Longer captions (30-60 chars) tend to perform better.")
        elif caption_length > 100:
            recommendations.append("Your caption is quite long. Consider making it more concise while keeping key information.")
        else:
            recommendations.append("Your caption length contributes positively ✍️")
        
        if hashtag_count == 0:
            recommendations.append("Add 2-4 relevant hashtags to increase discoverability and reach.")
        elif hashtag_count < 3:
            recommendations.append("Consider adding a few more hashtags (3-5 total) for better visibility.")
        elif hashtag_count > 10:
            recommendations.append("You're using many hashtags. Focus on 3-5 most relevant ones for better results.")
        else:
            recommendations.append("Your hashtag strategy is working well 💯")
        
        if fft_max_freq > 500:
            recommendations.append("Your audio matches trending high-energy tracks. This aligns with high-performing content! 🎧")
        elif fft_max_freq < 100:
            recommendations.append("Consider using more energetic audio to match trending content patterns.")
        
        if fft_max_amp > 100000000:
            recommendations.append("Strong audio energy detected. This contributes positively 🔥")
        
        # Music-specific tips
        if prob_high < 0.4:
            recommendations.append("For music covers, try 7-10 second intros. Hook speed in first 1-2 seconds is crucial ✂️")
        
    else:
        # GRWM-specific recommendations
        if caption_length < 20:
            recommendations.append("Try adding more context or storytelling to your caption. GRWM viewers love personal touches ✨")
        elif caption_length > 100:
            recommendations.append("Your caption is quite long. Consider making it more concise while keeping key information.")
        else:
            recommendations.append("Your caption tone is on point 💅")
        
        if hashtag_count == 0:
            recommendations.append("Add 2-4 lifestyle/aesthetic hashtags to increase discoverability.")
        elif hashtag_count < 3:
            recommendations.append("Consider adding a few more hashtags (3-5 total) focused on lifestyle/aesthetic tags.")
        elif hashtag_count > 10:
            recommendations.append("You're using many hashtags. Focus on 3-5 most relevant lifestyle/aesthetic ones.")
        else:
            recommendations.append("Hashtag diversity is good 📱")
        
        if fft_max_freq > 500:
            recommendations.append("Audio vibe matches aesthetic content. This aligns with high-performing GRWM videos! 🎵")
        elif fft_max_freq < 100:
            recommendations.append("Consider using brighter, more energetic audio to match aesthetic vibes.")
        
        if fft_max_amp > 100000000:
            recommendations.append("Audio presence is strong. This contributes positively 🎧")
        
        # GRWM-specific tips
        if prob_high < 0.4:
            recommendations.append("Consider tightening your opening 1.2 seconds. Intro drag can hurt engagement ✂️")
            recommendations.append("Lighting could be brighter. Aesthetic consistency matters for GRWM content 💡")
    
    # Probability-based recommendations
    if prob_high >= 0.7:
        recommendations.append("Your video has strong potential! Keep up the great work 🎉")
    elif prob_high < 0.4:
        if is_music:
            recommendations.append("Focus on improving caption quality and audio clarity to boost performance.")
        else:
            recommendations.append("Focus on pacing changes and caption optimization to boost performance.")
    
    # Feature-specific recommendations
    positive_feature_names = [f['feature'] for f in positive_features[:3]]
    if 'caption_length' in positive_feature_names:
        recommendations.append("Caption strength is a key strength. Maintain this quality in future posts.")
    if 'hashtag_count' in positive_feature_names:
        recommendations.append("Hashtag strategy is working well. Continue using this approach.")
    
    negative_feature_names = [f['feature'] for f in negative_features[:3]]
    if 'hashtag_count' in negative_feature_names:
        if is_music:
            recommendations.append("Optimize hashtag selection. Use trending, music-specific tags.")
        else:
            recommendations.append("Hashtags not focused enough. Use lifestyle/aesthetic tags.")
    
    return recommendations[:6]  # Limit to 6 recommendations


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
