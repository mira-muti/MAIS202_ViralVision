"""
Heuristic scoring engine for music videos (MUSIC_ENGINE).

This module does not use ML. It scores music videos based on audio
features and simple metadata, and returns a detailed breakdown.
"""

from typing import Dict, List, Tuple


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _scale_feature(x: float, low: float, high: float) -> float:
    """
    Linearly scale a value x from [low, high] into [0, 100].
    Values outside the range are clipped.
    """
    if high == low:
        return 0.0
    return _clamp((x - low) / (high - low) * 100.0)


def score_music_video(
    audio_features: Dict[str, float],
    caption_length: int,
    hashtag_count: int,
) -> Dict:
    """
    Score a music video using heuristic rules (no ML).

    Args:
        audio_features: Dict with keys like rms_energy, spectral_centroid,
                        spectral_rolloff, zcr, dynamic_range
        caption_length: Length of the caption text
        hashtag_count: Number of hashtags
    """
    strengths: List[str] = []
    improvements: List[str] = []

    rms = float(audio_features.get("rms_energy", 0.0) or 0.0)
    centroid = float(audio_features.get("spectral_centroid", 0.0) or 0.0)
    rolloff = float(audio_features.get("spectral_rolloff", 0.0) or 0.0)
    zcr = float(audio_features.get("zcr", 0.0) or 0.0)
    dyn_range = float(audio_features.get("dynamic_range", 0.0) or 0.0)

    # Heuristic ranges (roughly tuned for TikTok-style audio)
    audio_quality_score = _clamp(
        0.6 * _scale_feature(rms, 0.02, 0.2)
        + 0.4 * _scale_feature(dyn_range, 0.1, 0.8)
    )

    hook_strength_score = _clamp(
        0.6 * _scale_feature(centroid, 1500.0, 4500.0)
        + 0.4 * _scale_feature(rms, 0.03, 0.25)
    )

    clarity_score = _clamp(
        0.6 * (100.0 - _scale_feature(zcr, 0.02, 0.2))  # lower ZCR is cleaner
        + 0.4 * _scale_feature(rolloff, 2000.0, 8000.0)
    )

    production_score = _clamp(
        0.5 * _scale_feature(dyn_range, 0.1, 1.0)
        + 0.5 * _scale_feature(rolloff, 3000.0, 9000.0)
    )

    # Metadata score
    if caption_length <= 0:
        caption_score = 20.0
        improvements.append(
            "Add a caption – it helps viewers connect with the story behind your song."
        )
    elif caption_length < 20:
        caption_score = 60.0
        improvements.append(
            "Caption is quite short – consider adding more context or emotion."
        )
    elif caption_length > 120:
        caption_score = 70.0
        improvements.append(
            "Caption is long – try making it more concise while keeping the hook."
        )
    else:
        caption_score = 90.0
        strengths.append("Caption length feels balanced for music content.")

    if hashtag_count == 0:
        hashtag_score = 30.0
        improvements.append(
            "Add 2–4 relevant music hashtags to increase discoverability."
        )
    elif 1 <= hashtag_count <= 6:
        hashtag_score = 90.0
        strengths.append("Hashtag count looks strong for music discovery.")
    elif hashtag_count > 12:
        hashtag_score = 60.0
        improvements.append(
            "You have many hashtags – focus on 3–6 highly relevant music tags."
        )
    else:
        hashtag_score = 80.0

    metadata_score = _clamp(0.6 * caption_score + 0.4 * hashtag_score)

    # Final virality score (music-specific)
    final_music_score = _clamp(
        0.45 * audio_quality_score
        + 0.25 * hook_strength_score
        + 0.15 * clarity_score
        + 0.15 * metadata_score
    )

    # Strength messages from scores
    if audio_quality_score > 75:
        strengths.append("Your audio quality is strong – great loudness and dynamics.")
    elif audio_quality_score < 50:
        improvements.append(
            "Overall audio quality could be improved – consider cleaner recording or mastering."
        )

    if hook_strength_score > 75:
        strengths.append("Your intro hook feels strong – great for catching attention.")
    elif hook_strength_score < 50:
        improvements.append(
            "Hook energy seems low – consider making the first seconds more impactful."
        )

    if clarity_score > 75:
        strengths.append("Your audio clarity is strong – important details are easy to hear.")
    elif clarity_score < 50:
        improvements.append(
            "Brightness and clarity are on the lower side – try EQ or cleaner instruments/vocals."
        )

    if production_score > 75:
        strengths.append("Production feels polished with good dynamics and high-frequency detail.")
    elif production_score < 50:
        improvements.append(
            "Production could be enhanced – experiment with mixing to bring out key elements."
        )

    # Ensure we always return at least one message
    if not strengths:
        strengths.append("Your track has solid potential – keep iterating and releasing.")
    if not improvements:
        improvements.append(
            "Small tweaks to mix, hook, or metadata could further boost performance."
        )

    return {
        "engine": "music",
        "audio_quality_score": audio_quality_score,
        "hook_strength_score": hook_strength_score,
        "clarity_score": clarity_score,
        "production_score": production_score,
        "metadata_score": metadata_score,
        "final_music_score": final_music_score,
        "strengths": strengths,
        "improvements": improvements,
    }


