"""
File: audio_features.py

Owner: Backend Pod – Person 1

Purpose: 
    Extract audio features from video audio tracks using librosa.
    Works on 5-second audio clips extracted via ffmpeg.
    
Functions to implement:
    - load_audio_first5(audio_path, sr): Load first 5 seconds of audio
    - compute_rms(y): Root mean square energy
    - estimate_tempo(y, sr): Beats per minute
    - spectral_centroid(y, sr): Frequency center of mass
    - zero_crossing_rate(y): Rate of sign changes
    - mfcc_stats(y, sr, n): Mean MFCCs (1-13)
    - extract_audio_features(audio_path): Main function returning feature dict

Collaboration Rules:
    - Only Person 1 edits this file.
    - Output dictionary keys must match column names in audio_features.csv.
    - Handle errors gracefully (missing files, corrupted audio).

Dependencies:
    - librosa (audio processing)
    - numpy (numerical operations)
"""

import numpy as np
# import librosa
from typing import Dict
from pathlib import Path


def load_audio_first5(audio_path: str, sr: int = 44100) -> np.ndarray:
    """
    Load the first 5 seconds of audio from a WAV file.
    
    Args:
        audio_path (str): Path to audio file (.wav)
        sr (int): Sample rate (default 44100 Hz)
        
    Returns:
        np.ndarray: Audio time series (mono)
        
    Notes:
        - Audio should already be 5s from ffmpeg extraction
        - Returns mono audio (single channel)
        - Raises exception if file doesn't exist
    """
    pass


def compute_rms(y: np.ndarray) -> float:
    """
    Compute root mean square (RMS) energy of audio signal.
    
    Args:
        y (np.ndarray): Audio time series
        
    Returns:
        float: RMS energy value (typically 0-1)
        
    Notes:
        - Indicates overall loudness/energy of the audio
        - Higher values = louder audio
    """
    pass


def estimate_tempo(y: np.ndarray, sr: int) -> float:
    """
    Estimate tempo (beats per minute) using librosa.
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        
    Returns:
        float: Estimated tempo in BPM (typically 60-180)
        
    Notes:
        - Uses librosa.beat.tempo()
        - Returns 0 if tempo detection fails
    """
    pass


def spectral_centroid(y: np.ndarray, sr: int) -> float:
    """
    Compute mean spectral centroid (brightness of sound).
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        
    Returns:
        float: Mean spectral centroid in Hz
        
    Notes:
        - Higher values = brighter, higher-frequency content
        - Useful for distinguishing music from speech
    """
    pass


def zero_crossing_rate(y: np.ndarray) -> float:
    """
    Compute zero-crossing rate (sign changes per second).
    
    Args:
        y (np.ndarray): Audio time series
        
    Returns:
        float: Mean zero-crossing rate (0-1)
        
    Notes:
        - Higher for noisy/percussive sounds
        - Lower for tonal/pitched sounds
    """
    pass


def mfcc_stats(y: np.ndarray, sr: int, n: int = 13) -> Dict[str, float]:
    """
    Compute mean of first n MFCCs (Mel-frequency cepstral coefficients).
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        n (int): Number of MFCCs to compute (default 13)
        
    Returns:
        Dict[str, float]: Dictionary with keys "mfcc1", "mfcc2", ..., "mfcc{n}"
        
    Notes:
        - MFCCs capture timbral texture of audio
        - Return mean across all frames
        - For 2-week project, use n=5 for simplicity
    """
    pass


def extract_audio_features(audio_path: str) -> Dict[str, float]:
    """
    Extract all audio features from a single audio file.
    
    Args:
        audio_path (str): Path to .wav file
        
    Returns:
        Dict[str, float]: Dictionary containing:
            - rms: Root mean square energy
            - tempo_bpm: Estimated tempo
            - spectral_centroid: Mean spectral centroid
            - zcr: Zero-crossing rate
            - mfcc1, mfcc2, ..., mfcc5: Mean MFCCs
            
    Notes:
        - This is the main function called by preprocess.py
        - Returns None or empty dict if extraction fails
        - Log errors but don't crash the pipeline
        
    Example:
        >>> features = extract_audio_features('media/audio/7239401123.wav')
        >>> print(features)
        {'rms': 0.23, 'tempo_bpm': 128.5, 'spectral_centroid': 2150.3, ...}
    """
    pass


if __name__ == "__main__":
    # Example usage (to be implemented)
    # sample_audio = "media/audio/test_video.wav"
    # features = extract_audio_features(sample_audio)
    # print(features)
    pass

