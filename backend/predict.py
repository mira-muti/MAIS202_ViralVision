"""
Prediction dispatcher for ViralVision.

Selects between:
- GRWM_ENGINE (ML model) for non-music niches
- MUSIC_ENGINE (heuristic) for music niche
"""

import json
from typing import Dict

from extract_audio_features import extract_audio_feature_dict
from grwm_engine import predict_grwm_video
from music_engine import score_music_video
from prediction_logger import log_prediction


def predict_video(
    video_path: str,
    title: str,
    hashtags: str,
    niche: str,
    video_filename: str | None = None,
) -> Dict:
    """
    Dispatch prediction to the appropriate engine based on niche.
    """
    niche_lower = (niche or "").lower()

    # Common features used by both engines
    audio_features = extract_audio_feature_dict(video_path, seconds=5)

    caption_length = len(title.strip()) if title else 0
    hashtag_count = 0
    if hashtags:
        hashtag_tokens = hashtags.split()
        hashtag_count = sum(1 for token in hashtag_tokens if token.startswith("#"))

    if niche_lower == "music":
        # MUSIC_ENGINE (heuristic only)
        music_result = score_music_video(audio_features, caption_length, hashtag_count)
        score = music_result["final_music_score"]
        probability = score / 100.0
        label = "High" if probability >= 0.6 else "Low"

        result: Dict = {
            "engine": "music",
            "score": score,
            "label": label,
            "probability": probability,
            "features_positive": music_result["strengths"],
            "features_negative": music_result["improvements"],
            "raw_features": {
                "audio": audio_features,
                "text": {
                    "caption_length": caption_length,
                    "hashtag_count": hashtag_count,
                    "niche": niche,
                },
            },
            # Legacy fields for frontend compatibility
            "prob_high": probability,
            "prob_low": 1.0 - probability,
            "top_positive_features": [],
            "top_negative_features": [],
            "recommendations": music_result["improvements"],
        }
    else:
        # GRWM_ENGINE (ML)
        grwm_result = predict_grwm_video(
            video_path=video_path,
            title=title,
            hashtags=hashtags,
            niche=niche,
        )
        # Add engine + score field and raw audio/text features for completeness
        result = {
            **grwm_result,
            "engine": "grwm",
        }

    # Log basic info
    if video_filename:
        log_payload = {
            "label": result.get("label", "Unknown"),
            "prob_high": result.get("prob_high", result.get("probability", 0.0)),
            "prob_low": result.get("prob_low", 1.0 - result.get("prob_high", 0.0)),
        }
        log_prediction(video_filename, title, niche, log_payload)

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

    res = predict_video(video_path, title, hashtags, niche)
    print(json.dumps(res, indent=2))
