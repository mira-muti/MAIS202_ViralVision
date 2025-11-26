"""
Frame utility functions for video processing.

What this module does:
- Extracts frames from video clips at specified FPS
- Converts frames to grayscale for processing
- Provides utilities for frame manipulation

How the new feature works:
- Uses MoviePy VideoFileClip to extract frames at regular intervals
- Converts RGB frames to BGR for OpenCV compatibility
- Handles frame extraction errors gracefully

What changed from the old Mediapipe version:
- No changes needed - this module was already MediaPipe-free
- Still uses MoviePy for frame extraction
"""

import cv2
import numpy as np
from typing import List
from moviepy import VideoFileClip


def extract_frames_from_clip(clip, max_seconds: float = 3.0, fps: int = 8) -> List[np.ndarray]:
    """
    Extract frames from a video clip at specified FPS.
    
    Args:
        clip: VideoFileClip object
        max_seconds: Maximum seconds to extract
        fps: Target frames per second
        
    Returns:
        List of frame arrays (BGR format for OpenCV)
    """
    frames = []
    frame_duration = 1.0 / fps
    duration = min(max_seconds, clip.duration)
    
    t = 0.0
    while t < duration:
        try:
            frame = clip.get_frame(t)
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame_bgr)
        except Exception:
            break
        t += frame_duration
    
    return frames


def frames_to_grayscale(frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Convert frames to grayscale.
    
    Args:
        frames: List of BGR frames
        
    Returns:
        List of grayscale frames
    """
    return [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
