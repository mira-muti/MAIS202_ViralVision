"""
File: preprocess.py

Owner: Backend Pod – Person 1

Purpose: 
    Glue module for batch feature extraction pipeline.
    Orchestrates audio and visual feature extraction across all videos.
    Calls functions from audio_features.py and visual_features.py.

Functions to implement:
    - extract_all_audio_features(meta_csv, audio_dir): Batch audio extraction
    - extract_all_visual_features(meta_csv, frames_root): Batch visual extraction
    - merge_features(meta_csv, audio_csv, visual_csv): Merge into one CSV

Collaboration Rules:
    - Only Person 1 edits this file.
    - Imports from audio_features.py and visual_features.py
    - Output CSVs go to data/processed/
    - Coordinate with Person 2 on feature column names

Dependencies:
    - pandas (CSV handling)
    - audio_features (audio extraction module)
    - visual_features (visual extraction module)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# from audio_features import extract_audio_features
# from visual_features import extract_visual_features


def extract_all_audio_features(meta_csv: str, audio_dir: str) -> Path:
    """
    Extract audio features for all videos in metadata.
    
    Args:
        meta_csv (str): Path to metadata.csv
        audio_dir (str): Directory containing .wav files
        
    Returns:
        Path: Path to saved audio_features.csv
        
    Notes:
        - Reads metadata.csv for video_id list
        - Calls extract_audio_features() from audio_features.py for each video
        - Saves to data/processed/audio_features.csv
        - Logs errors for videos that fail
    """
    pass


def extract_all_visual_features(meta_csv: str, frames_root: str) -> Path:
    """
    Extract visual features for all videos in metadata.
    
    Args:
        meta_csv (str): Path to metadata.csv
        frames_root (str): Root directory containing frame subdirectories
        
    Returns:
        Path: Path to saved visual_features.csv
        
    Notes:
        - Reads metadata.csv for video_id list
        - Calls extract_visual_features() from visual_features.py for each video
        - Saves to data/processed/visual_features.csv
        - Logs errors for videos that fail
    """
    pass


def merge_features(meta_csv: str, audio_csv: str, visual_csv: str) -> Path:
    """
    Merge metadata, audio features, and visual features into one dataset.
    
    Args:
        meta_csv (str): Path to metadata.csv
        audio_csv (str): Path to audio_features.csv
        visual_csv (str): Path to visual_features.csv
        
    Returns:
        Path: Path to saved features_merged.csv
        
    Notes:
        - Merges on video_id column
        - Keeps engagement columns (likes, views, comments, shares, niche)
        - Saves to data/processed/features_merged.csv
        - Handles missing features gracefully (some videos may fail extraction)
    """
    pass


if __name__ == "__main__":
    # Example usage (to be implemented)
    # Run via scripts/04_build_dataset.py instead
    # audio_csv = extract_all_audio_features('data/raw/metadata.csv', 'media/audio')
    # visual_csv = extract_all_visual_features('data/raw/metadata.csv', 'media/frames')
    # merged = merge_features('data/raw/metadata.csv', audio_csv, visual_csv)
    pass

