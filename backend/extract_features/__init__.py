"""
Feature extraction module for ViralVision.

This module provides unified audio and visual feature extraction
from video files for engagement prediction.
"""

from .video_downloader import download_video, is_local_file
from .audio_features import extract_audio_features
from .visual_features import extract_visual_features

__all__ = [
    'download_video',
    'is_local_file',
    'extract_audio_features',
    'extract_visual_features',
]

