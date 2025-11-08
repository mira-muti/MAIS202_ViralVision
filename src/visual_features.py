"""
File: visual_features.py

Owner: Backend Pod – Person 1

Purpose: 
    Extract visual features from video frames using OpenCV.
    Works on 5 frames sampled at 1 fps from the first 5 seconds.
    
Functions to implement:
    - load_frames(folder): Load all frames from a video's frame directory
    - avg_brightness(frames): Average brightness across frames
    - contrast(frames): Contrast measure
    - color_variance(frames): Variance in color distribution
    - motion_score(frames): Motion between consecutive frames
    - face_count(frames): Number of faces detected (optional)
    - extract_visual_features(frames_folder): Main function returning feature dict

Collaboration Rules:
    - Only Person 1 edits this file.
    - Output dictionary keys must match column names in visual_features.csv.
    - Handle missing frames gracefully (some videos may have <5 frames).

Dependencies:
    - opencv-python (cv2)
    - numpy (numerical operations)
"""

import numpy as np
# import cv2
from typing import List, Dict
from pathlib import Path


def load_frames(folder: str) -> List[np.ndarray]:
    """
    Load all frame images from a video's frame directory.
    
    Args:
        folder (str): Path to frames directory (e.g., media/frames/7239401123/)
        
    Returns:
        List[np.ndarray]: List of frame images (BGR format)
        
    Notes:
        - Expects 5 frames named frame_001.jpg, frame_002.jpg, etc.
        - Returns empty list if no frames found
        - Frames should be sorted by filename
    """
    pass


def avg_brightness(frames: List[np.ndarray]) -> float:
    """
    Compute average brightness across all frames.
    
    Args:
        frames (List[np.ndarray]): List of frame images
        
    Returns:
        float: Mean brightness value (0-255)
        
    Notes:
        - Convert to grayscale and compute mean intensity
        - Average across all frames
        - Higher values = brighter video
    """
    pass


def contrast(frames: List[np.ndarray]) -> float:
    """
    Compute contrast measure across all frames.
    
    Args:
        frames (List[np.ndarray]): List of frame images
        
    Returns:
        float: Mean contrast (standard deviation of pixel intensities)
        
    Notes:
        - Higher values = more contrast/dynamic range
        - Can use std of grayscale intensities
    """
    pass


def color_variance(frames: List[np.ndarray]) -> float:
    """
    Compute variance in color distribution across frames.
    
    Args:
        frames (List[np.ndarray]): List of frame images
        
    Returns:
        float: Color variance metric
        
    Notes:
        - Measure how much colors change across frames
        - Can use variance of mean RGB values
        - Higher values = more colorful/dynamic video
    """
    pass


def motion_score(frames: List[np.ndarray]) -> float:
    """
    Compute motion score based on frame-to-frame differences.
    
    Args:
        frames (List[np.ndarray]): List of frame images
        
    Returns:
        float: Motion score (mean difference between consecutive frames)
        
    Notes:
        - Compute absolute difference between consecutive frames
        - Average across all frame pairs
        - Higher values = more motion/action
    """
    pass


def face_count(frames: List[np.ndarray]) -> int:
    """
    Count average number of faces detected across frames.
    
    Args:
        frames (List[np.ndarray]): List of frame images
        
    Returns:
        int: Average face count (rounded)
        
    Notes:
        - Use Haar Cascade classifier (cv2.CascadeClassifier)
        - Optional feature - return 0 if detection fails
        - Average face count across all frames
        - Useful for identifying talking-head vs. landscape videos
    """
    pass


def extract_visual_features(frames_folder: str) -> Dict[str, float]:
    """
    Extract all visual features from a video's frame directory.
    
    Args:
        frames_folder (str): Path to directory containing frames
        
    Returns:
        Dict[str, float]: Dictionary containing:
            - avg_brightness: Mean brightness (0-255)
            - contrast: Contrast measure
            - color_var: Color variance
            - motion_score: Motion between frames
            - faces: Average face count (optional)
            
    Notes:
        - This is the main function called by preprocess.py
        - Returns None or empty dict if extraction fails
        - Log errors but don't crash the pipeline
        
    Example:
        >>> features = extract_visual_features('media/frames/7239401123')
        >>> print(features)
        {'avg_brightness': 128.5, 'contrast': 45.2, 'color_var': 23.1, ...}
    """
    pass


if __name__ == "__main__":
    # Example usage (to be implemented)
    # sample_frames = "media/frames/test_video"
    # features = extract_visual_features(sample_frames)
    # print(features)
    pass

