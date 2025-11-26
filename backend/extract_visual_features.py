"""
Visual feature extraction module for ViralVision.

This module extracts visual features from video files using OpenCV.
Features include brightness, color variance, and motion intensity,
which can be used for explainability and future model improvements.
"""

import cv2
import numpy as np
from typing import Dict
from pathlib import Path


def extract_visual_features(
    video_path: str,
    max_seconds: int = 5,
    frame_sample_rate: int = 10,
) -> Dict[str, float]:
    """
    Extract visual features from the first few seconds of a video.
    
    This function samples frames from the beginning of the video and computes:
    - Average brightness across sampled frames
    - Average color variance across sampled frames
    - Motion intensity using optical flow between consecutive frames
    
    Args:
        video_path: Path to the video file (.mp4, .mov, etc.)
        max_seconds: Maximum number of seconds to process from the start (default: 5)
        frame_sample_rate: Approximate number of frames to sample per second (default: 10)
        
    Returns:
        Dictionary containing:
        - avg_brightness: Average brightness (0-255) across sampled frames
        - avg_color_variance: Average color variance across sampled frames
        - motion_intensity: Average magnitude of optical flow vectors
        - frame_count_used: Number of frames actually processed
        
    Raises:
        RuntimeError: If the video cannot be opened or processed
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise RuntimeError(f"Video file not found: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")
    
    try:
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Determine how many seconds to process
        seconds_to_process = min(max_seconds, duration)
        max_frame_index = int(seconds_to_process * fps) if fps > 0 else total_frames
        
        # Calculate frame step (sample every Nth frame)
        # If we want ~frame_sample_rate frames per second, step = fps / frame_sample_rate
        if fps > 0:
            frame_step = max(1, int(fps / frame_sample_rate))
        else:
            frame_step = 1
        
        # Initialize accumulators
        brightness_sum = 0.0
        color_variance_sum = 0.0
        motion_magnitudes = []
        frame_count = 0
        
        # Store previous frame for optical flow
        prev_gray = None
        prev_frame_idx = -1
        
        frame_idx = 0
        while frame_idx < max_frame_index:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process frames at the sample rate
            if frame_idx % frame_step == 0:
                # Convert to grayscale for brightness and optical flow
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 1) Compute brightness (mean pixel value in grayscale)
                brightness = np.mean(gray)
                brightness_sum += brightness
                
                # 2) Compute color variance
                # Compute variance across all 3 color channels
                frame_float = frame.astype(np.float32)
                color_variance = np.var(frame_float)
                color_variance_sum += color_variance
                
                # 3) Compute motion intensity using optical flow
                if prev_gray is not None:
                    # Calculate optical flow between previous and current frame
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray,
                        gray,
                        None,
                        pyr_scale=0.5,
                        levels=3,
                        winsize=15,
                        iterations=3,
                        poly_n=5,
                        poly_sigma=1.2,
                        flags=0
                    )
                    
                    # Compute magnitude of flow vectors
                    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                    mean_magnitude = np.mean(magnitude)
                    motion_magnitudes.append(mean_magnitude)
                
                # Update previous frame for next iteration
                prev_gray = gray
                prev_frame_idx = frame_idx
                frame_count += 1
            
            frame_idx += 1
        
        # Compute averages
        if frame_count > 0:
            avg_brightness = brightness_sum / frame_count
            avg_color_variance = color_variance_sum / frame_count
            motion_intensity = np.mean(motion_magnitudes) if motion_magnitudes else 0.0
        else:
            # No frames processed
            avg_brightness = 0.0
            avg_color_variance = 0.0
            motion_intensity = 0.0
        
        return {
            "avg_brightness": float(avg_brightness),
            "avg_color_variance": float(avg_color_variance),
            "motion_intensity": float(motion_intensity),
            "frame_count_used": frame_count
        }
        
    finally:
        cap.release()

