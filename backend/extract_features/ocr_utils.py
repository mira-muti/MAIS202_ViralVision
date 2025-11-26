"""
OCR utilities for text overlay detection.

What this module does:
- Detects text overlays in video frames using Tesseract OCR
- Provides text detection for feature extraction pipeline
- Returns binary presence indicator (1 if text found, 0 otherwise)

How the new feature works:
- Uses pytesseract to extract text from frames
- Checks first 40 frames for text longer than specified minimum length
- Returns True if any frame contains sufficient text, False otherwise

What changed from the old Mediapipe version:
- No changes needed - this module was already MediaPipe-free
- Updated to support checking first N frames with configurable text length threshold
"""

import cv2
import numpy as np
from typing import List

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available. Text detection will be disabled.")


def detect_text_in_frame(frame: np.ndarray, min_text_length: int = 4) -> bool:
    """
    Detect if text is present in a frame using Tesseract OCR.
    
    Args:
        frame: BGR frame array
        min_text_length: Minimum characters to consider as text (default: 4)
        
    Returns:
        True if text detected, False otherwise
    """
    if not TESSERACT_AVAILABLE:
        return False
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast for better OCR
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
        
        # Run OCR
        text = pytesseract.image_to_string(gray, config='--psm 6')
        text = text.strip()
        
        return len(text) >= min_text_length
    except Exception:
        return False


def detect_text_overlay(frames: List[np.ndarray], min_text_length: int = 4, max_frames: int = 40) -> bool:
    """
    Detect if text overlay is present in first N frames.
    
    Args:
        frames: List of BGR frames
        min_text_length: Minimum characters to consider as text (default: 4)
        max_frames: Maximum number of frames to check (default: 40)
        
    Returns:
        True if text detected in any of first N frames, False otherwise
    """
    if not frames:
        return False
    
    # Check first N frames (up to max_frames)
    check_frames = frames[:min(max_frames, len(frames))]
    
    for frame in check_frames:
        if detect_text_in_frame(frame, min_text_length=min_text_length):
            return True
    
    return False
