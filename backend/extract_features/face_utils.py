"""
Face detection utilities using OpenCV Haar Cascade.

What this module does:
- Detects faces in video frames using OpenCV's built-in Haar Cascade classifier
- Provides face detection for the first second of video
- Returns binary presence indicator (1 if face found, 0 otherwise)

How the new feature works:
- Uses cv2.CascadeClassifier with haarcascade_frontalface_default.xml
- Loads the cascade from cv2.data.haarcascades (built into OpenCV)
- Detects faces in grayscale frames for efficiency
- Returns 1 if any face is detected in the first second, else 0

What changed from the old Mediapipe version:
- REMOVED: MediaPipe face detection (mp.solutions.face_detection)
- REPLACED WITH: OpenCV Haar Cascade classifier
- BENEFIT: No external dependencies, works on Python 3.13
- SIMPLER: Binary output (face present/not present) instead of bounding boxes
"""

import cv2
import numpy as np
from typing import List


class FaceDetector:
    """OpenCV Haar Cascade face detector wrapper."""
    
    def __init__(self):
        """
        Initialize face detector using OpenCV Haar Cascade.
        
        Uses the built-in frontal face cascade classifier that comes with OpenCV.
        """
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")
    
    def detect_faces(self, frame: np.ndarray) -> List:
        """
        Detect faces in a frame using Haar Cascade.
        
        Args:
            frame: BGR frame array or grayscale frame
            
        Returns:
            List of (x, y, width, height) bounding boxes
        """
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            return faces.tolist() if len(faces) > 0 else []
        except Exception:
            return []
    
    def has_face(self, frame: np.ndarray) -> bool:
        """
        Check if frame contains at least one face.
        
        Args:
            frame: BGR frame array or grayscale frame
            
        Returns:
            True if face detected, False otherwise
        """
        return len(self.detect_faces(frame)) > 0
    
    def detect_face_in_first_second(self, frames: List[np.ndarray], fps: int = 8) -> bool:
        """
        Detect if any face is present in the first second of video.
        
        Args:
            frames: List of BGR frames
            fps: Frames per second (to determine first second)
            
        Returns:
            True if face detected in first second, False otherwise
        """
        if not frames:
            return False
        
        # Check frames in first second (approximately fps frames)
        first_second_frames = frames[:min(fps, len(frames))]
        
        for frame in first_second_frames:
            if self.has_face(frame):
                return True
        
        return False
