"""
Audio feature extraction module for ViralVision.

This module extracts audio features from video files using MoviePy and Librosa.
It provides both basic FFT features (for model compatibility) and extended
audio analysis features for future improvements, including a music-specific
feature set.
"""

import numpy as np
import librosa
from moviepy import VideoFileClip
from typing import Tuple, Dict
from pathlib import Path


def load_audio_segment(video_path: str, seconds: int = 5) -> Tuple[np.ndarray, int]:
    """
    Load the first N seconds of audio from a video file.
    
    This function extracts audio from a video, converts it to mono, and returns
    the audio samples along with the sample rate. Only the first 'seconds' of
    audio are extracted.
    
    Args:
        video_path: Path to the video file (.mp4, .mov, etc.)
        seconds: Number of seconds to extract (default: 5)
        
    Returns:
        Tuple of (audio_samples, sample_rate) where:
        - audio_samples: 1D numpy array of audio samples (mono)
        - sample_rate: Sample rate in Hz (integer)
        
    Raises:
        FileNotFoundError: If the video file doesn't exist
        RuntimeError: If the video has no audio track or extraction fails
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    clip = None
    try:
        # Load video
        clip = VideoFileClip(str(video_path))

        # Check if video has audio
        if clip.audio is None:
            clip.close()
            raise RuntimeError(f"Video has no audio track: {video_path}")

        # Determine how many seconds to extract (don't exceed video duration)
        duration_to_extract = min(seconds, clip.duration)

        # Get audio samples from full audio and slice to desired duration
        audio = clip.audio
        audio_fps = audio.fps

        # Convert entire audio to samples
        samples = audio.to_soundarray(fps=audio_fps)

        # Compute number of samples for the desired duration
        num_samples = int(duration_to_extract * audio_fps)
        if len(samples) > num_samples:
            samples = samples[:num_samples]

        # Convert to mono
        if samples.ndim > 1:
            mono = samples.mean(axis=1)
        else:
            mono = samples

        y = mono.astype(np.float32)
        sr = int(audio_fps)

        clip.close()
        return y, sr

    except Exception as e:
        if clip is not None:
            clip.close()
        if isinstance(e, (FileNotFoundError, RuntimeError)):
            raise
        raise RuntimeError(f"Error extracting audio from {video_path}: {str(e)}")


def extract_fft_features(video_path: str, seconds: int = 5) -> Tuple[float, float]:
    """
    Extract FFT-based features from the first N seconds of a video's audio.
    
    This function computes the Fast Fourier Transform of the audio and finds
    the frequency with the maximum amplitude. This matches the feature extraction
    used in the original model training.
    
    Args:
        video_path: Path to the video file
        seconds: Number of seconds to analyze (default: 5)
        
    Returns:
        Tuple of (fft_max_freq, fft_max_amp) where:
        - fft_max_freq: Frequency (in Hz) with maximum amplitude
        - fft_max_amp: Magnitude at that frequency
        
    Raises:
        FileNotFoundError: If the video file doesn't exist
        RuntimeError: If audio extraction fails
    """
    # Load audio segment
    y, sr = load_audio_segment(video_path, seconds)
    
    # Compute FFT (using real FFT for efficiency, only positive frequencies)
    fft_vals = np.abs(np.fft.rfft(y))
    fft_freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
    
    # Find the frequency bin with maximum magnitude
    max_idx = np.argmax(fft_vals)
    max_freq = float(fft_freqs[max_idx])
    max_amp = float(fft_vals[max_idx])
    
    return max_freq, max_amp


def extract_audio_feature_dict(video_path: str, seconds: int = 5) -> Dict[str, float]:
    """
    Extract comprehensive audio features from a video file.
    
    This function computes multiple audio features including RMS energy, zero-crossing
    rate, spectral features, tempo, and the original FFT features. This extended
    feature set can be used for future model improvements.
    
    Args:
        video_path: Path to the video file
        seconds: Number of seconds to analyze (default: 5)
        
    Returns:
        Dictionary containing:
        - rms_energy: Mean RMS (Root Mean Square) energy over the clip
        - zcr: Mean zero-crossing rate (rate of sign changes)
        - spectral_centroid: Mean spectral centroid (brightness measure)
        - spectral_rolloff: Mean spectral rolloff at 85% energy
        - tempo: Estimated tempo in beats per minute (BPM)
        - fft_max_freq: Frequency with maximum FFT amplitude (Hz)
        - fft_max_amp: Magnitude at maximum frequency
        
    Raises:
        FileNotFoundError: If the video file doesn't exist
        RuntimeError: If audio extraction fails
    """
    # Load audio segment
    y, sr = load_audio_segment(video_path, seconds)
    
    # Compute FFT features (for model compatibility)
    fft_max_freq, fft_max_amp = extract_fft_features(video_path, seconds)
    
    # Compute RMS energy (average energy level)
    rms = librosa.feature.rms(y=y)[0]
    rms_energy = float(np.mean(rms))
    
    # Compute zero-crossing rate (rate of sign changes, indicates noisiness)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = float(np.mean(zcr))
    
    # Compute spectral centroid (brightness measure, weighted mean frequency)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_centroid = float(np.mean(spectral_centroids))
    
    # Compute spectral rolloff (frequency below which 85% of energy is contained)
    try:
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, roll_percent=0.85
        )[0]
        spectral_rolloff_mean = float(np.mean(spectral_rolloff))
    except TypeError:
        # Fallback for older librosa versions that may use a different signature
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_rolloff_mean = float(np.mean(spectral_rolloff))
    
    # Dynamic range (max - min amplitude)
    dynamic_range = float(np.max(y) - np.min(y)) if len(y) > 0 else 0.0
    
    return {
        "rms_energy": rms_energy,
        "zcr": zcr_mean,
        "spectral_centroid": spectral_centroid,
        "spectral_rolloff": spectral_rolloff_mean,
        "dynamic_range": dynamic_range,
        "fft_max_freq": fft_max_freq,
        "fft_max_amp": fft_max_amp,
    }
