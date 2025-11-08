"""
File: utils.py

Owner: Backend Pod – Shared (All team members can use)

Purpose: 
    Utility functions shared across the project.
    This module contains helper functions for:
    - File path management
    - Data normalization
    - Logging and error handling
    - Common data transformations

Functions to implement:
    - get_project_root(): Return the root directory path
    - get_data_path(filename): Build path to data files
    - get_model_path(filename): Build path to model files
    - normalize_features(df): Normalize/scale feature values
    - validate_video_file(video_path): Check if video file is valid
    - setup_logging(): Configure logging for the project

Collaboration Rules:
    - All team members can add utility functions here.
    - Document all functions clearly.
    - Avoid duplicating functionality.
    - Keep functions small and focused.

Dependencies:
    - os, pathlib (path operations)
    - logging (error tracking)
    - pandas, numpy (data operations)
"""

import os
from pathlib import Path
import logging
# import pandas as pd
# import numpy as np


def get_project_root():
    """
    Get the absolute path to the project root directory.
    
    Returns:
        Path: Path object pointing to project root
        
    Notes:
        - Useful for building absolute paths
        - Works regardless of where script is called from
    """
    pass


def get_data_path(filename):
    """
    Build path to a file in the data directory.
    
    Args:
        filename (str): Name of the file (e.g., 'final_dataset.csv')
        
    Returns:
        str: Full path to the data file
        
    Example:
        >>> get_data_path('final_dataset.csv')
        '/path/to/MAIS202_ViralVision/data/final_dataset.csv'
    """
    pass


def get_model_path(filename='model.pkl'):
    """
    Build path to a file in the models directory.
    
    Args:
        filename (str): Name of the model file
        
    Returns:
        str: Full path to the model file
    """
    pass


def normalize_features(df, method='standard'):
    """
    Normalize/scale feature values in a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing features
        method (str): Normalization method ('standard', 'minmax', 'robust')
        
    Returns:
        pd.DataFrame: Normalized DataFrame
        
    Notes:
        - Apply same normalization to train and test data
        - Save scaler parameters for consistency
    """
    pass


def validate_video_file(video_path):
    """
    Check if a video file is valid and readable.
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        bool: True if valid, False otherwise
        
    Notes:
        - Check file exists
        - Check file extension (.mp4, .avi, etc.)
        - Optionally verify file is not corrupted
    """
    pass


def setup_logging(log_level=logging.INFO):
    """
    Configure logging for the project.
    
    Args:
        log_level: Logging level (INFO, DEBUG, WARNING, ERROR)
        
    Notes:
        - Set up console and file logging
        - Create logs directory if needed
        - Use consistent format across all modules
    """
    pass


def create_directory_structure():
    """
    Create all necessary directories for the project.
    
    Creates:
        - data/raw/
        - data/processed/
        - models/
        - logs/
        
    Notes:
        - Safe to call multiple times (won't overwrite existing dirs)
        - Useful for initial project setup
    """
    pass


if __name__ == "__main__":
    # Example usage (to be implemented)
    # setup_logging()
    # create_directory_structure()
    pass

