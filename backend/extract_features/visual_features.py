"""
Visual feature extraction module for ViralVision.

What this module does:
- Extracts visual features from the first 3 seconds of video
- Uses OpenCV for image processing and face detection
- Uses Tesseract OCR for text overlay detection
- Returns a dictionary of normalized feature values

How the new features work:
1. avg_brightness: Mean pixel intensity over first 40 frames (or all available)
2. color_std_dev: Standard deviation of HSV color values
3. motion_intensity: Average absolute frame difference over first 40 frames
4. scene_change_rate: Histogram comparison (chi-square) to detect scene changes
5. face_present: OpenCV Haar Cascade detects faces in first second (1 if found, 0 otherwise)
6. text_overlay_present: Tesseract OCR checks first 40 frames for text > 4 chars (1 if found, 0 otherwise)
7. hue_entropy: Normalized histogram of hue channel (32 bins) - kept for compatibility

What changed from the old Mediapipe version:
- REMOVED: MediaPipe face detection dependency
- REPLACED: MediaPipe FaceDetector → OpenCV Haar Cascade
- UPDATED: All features now computed over first 40 frames (or first second for face detection)
- IMPROVED: More robust error handling and frame sampling
- BENEFIT: Works on Python 3.13, no external ML dependencies
"""

import cv2
import numpy as np
from typing import Dict
from pathlib import Path
from moviepy import VideoFileClip

from .frame_utils import extract_frames_from_clip, frames_to_grayscale
from .face_utils import FaceDetector
from .ocr_utils import detect_text_overlay


def extract_visual_features(video_path: str, max_seconds: float = 3.0) -> Dict[str, float]:
    """
    Extract comprehensive visual features from the first N seconds of a video.
    
    Args:
        video_path: Path to video file
        max_seconds: Maximum seconds to analyze (default: 3.0)
        
    Returns:
        Dictionary containing:
        - avg_brightness: Average brightness (0-255) over first 40 frames
        - color_std_dev: Standard deviation of HSV color values
        - motion_intensity: Average frame difference magnitude over first 40 frames
        - scene_change_rate: Scene change rate using histogram comparison
        - hue_entropy: Hue entropy (normalized histogram, 32 bins)
        - face_present: 1.0 if face detected in first second, 0.0 otherwise
        - text_overlay_present: 1.0 if text detected in first 40 frames, 0.0 otherwise
        
    Raises:
        RuntimeError: If video cannot be processed
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise RuntimeError(f"Video file not found: {video_path}")
    
    clip = None
    
    try:
        # Load video
        clip = VideoFileClip(str(video_path))
        duration = min(max_seconds, clip.duration)
        
        # Extract frames at 8 FPS (will get ~24 frames for 3 seconds)
        # We'll use up to 40 frames for feature computation
        frames = extract_frames_from_clip(clip, max_seconds=duration, fps=8)
        
        if not frames:
            raise RuntimeError("No frames extracted from video")
        
        # Limit to first 40 frames for consistent feature computation
        max_frames = min(40, len(frames))
        frames = frames[:max_frames]
        
        # Compute features
        
        # 1. Average brightness: mean pixel intensity over first 40 frames
        gray_frames = frames_to_grayscale(frames)
        brightness_values = [np.mean(frame) for frame in gray_frames]
        avg_brightness = float(np.mean(brightness_values))
        
        # 2. Color standard deviation: std dev of HSV values
        std_values = []
        for frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Compute std across all HSV channels
            std = np.std(hsv.astype(np.float32))
            std_values.append(std)
        color_std_dev = float(np.mean(std_values))
        
        # 3. Motion intensity: abs(frame[i] - frame[i-1]).mean() averaged over first 40 frames
        motion_magnitudes = []
        prev_gray = gray_frames[0]
        for gray in gray_frames[1:]:
            diff = cv2.absdiff(prev_gray, gray)
            magnitude = np.mean(diff)
            motion_magnitudes.append(magnitude)
            prev_gray = gray
        motion_intensity = float(np.mean(motion_magnitudes)) if motion_magnitudes else 0.0
        
        # 4. Scene change rate: histogram comparison (chi-square or correlation)
        scene_changes = []
        prev_hist = cv2.calcHist(
            [frames[0]], [0, 1, 2], None, [8, 8, 8],
            [0, 256, 0, 256, 0, 256]
        )
        prev_hist = cv2.normalize(prev_hist, prev_hist).flatten()
        
        for frame in frames[1:]:
            hist = cv2.calcHist(
                [frame], [0, 1, 2], None, [8, 8, 8],
                [0, 256, 0, 256, 0, 256]
            )
            hist = cv2.normalize(hist, hist).flatten()
            
            # Use chi-square for better scene change detection
            chi_square = cv2.compareHist(
                prev_hist.reshape(-1, 1), hist.reshape(-1, 1), cv2.HISTCMP_CHISQR
            )
            # Normalize chi-square to [0, 1] range (higher = more change)
            # Using a simple normalization: 1 / (1 + chi_square / 1000)
            normalized_change = 1.0 / (1.0 + chi_square / 1000.0)
            scene_changes.append(normalized_change)
            prev_hist = hist
        
        scene_change_rate = float(np.mean(scene_changes)) if scene_changes else 0.0
        
        # 5. Hue entropy: normalized histogram of hue channel (32 bins)
        entropy_values = []
        for frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0]
            
            # Compute histogram with 32 bins
            hist, _ = np.histogram(hue.flatten(), bins=32, range=(0, 180))
            hist = hist[hist > 0]  # Remove zeros
            if len(hist) > 0:
                hist = hist / hist.sum()  # Normalize
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                entropy_values.append(entropy)
        
        hue_entropy = float(np.mean(entropy_values)) if entropy_values else 0.0
        
        # 6. Face presence in first second: OpenCV Haar Cascade
        face_detector = FaceDetector()
        # Check first second (approximately 8 frames at 8 FPS)
        first_second_frames = frames[:min(8, len(frames))]
        face_present = 1.0 if face_detector.detect_face_in_first_second(first_second_frames, fps=8) else 0.0
        
        # 7. Text overlay presence: Tesseract OCR on first 40 frames
        # Check if text > 4 chars found in any of first 40 frames
        text_overlay_present = 1.0 if detect_text_overlay(frames, min_text_length=4, max_frames=40) else 0.0
        
        return {
            "avg_brightness": avg_brightness,
            "color_std_dev": color_std_dev,
            "motion_intensity": motion_intensity,
            "scene_change_rate": scene_change_rate,
            "hue_entropy": hue_entropy,
            "face_present": face_present,
            "text_overlay_present": text_overlay_present,
        }
        
    except Exception as e:
        raise RuntimeError(f"Error extracting visual features: {str(e)}")
    finally:
        if clip is not None:
            clip.close()
