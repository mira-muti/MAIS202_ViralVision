"""
File: labeler.py

Owner: Backend Pod – Person 2

Purpose: 
    Add High/Low performance labels to the feature dataset based on engagement metrics.
    Uses percentile-based labeling within each niche category.
    
Functions to implement:
    - compute_engagement_ratio(df): Calculate likes/views ratio
    - assign_labels_by_percentile(df, niche_col): Assign High/Low labels
    - add_labels_by_percentile(features_csv, niche_col): Main function
    
Labeling Logic:
    - engagement_ratio = likes / max(views, 1)
    - Within each niche, compute percentiles
    - High: >= 75th percentile
    - Low: <= 25th percentile
    - Middle 50% excluded for crisp classification

Collaboration Rules:
    - Only Person 2 edits this file.
    - Input: features_merged.csv with engagement columns
    - Output: final_dataset.csv with perf_label column

Dependencies:
    - pandas (data manipulation)
    - numpy (numerical operations)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def compute_engagement_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute engagement ratio (likes / views) for each video.
    
    Args:
        df (pd.DataFrame): DataFrame with 'likes' and 'views' columns
        
    Returns:
        pd.DataFrame: DataFrame with new 'engagement_ratio' column
        
    Notes:
        - engagement_ratio = likes / max(views, 1) to avoid division by zero
        - Values typically in range [0, 0.5] for most videos
        - Higher ratio = more engaging content
    """
    pass


def assign_labels_by_percentile(
    df: pd.DataFrame, 
    niche_col: str = "niche",
    percentile_high: float = 75.0,
    percentile_low: float = 25.0
) -> pd.DataFrame:
    """
    Assign High/Low labels based on niche-wise percentiles.
    
    Args:
        df (pd.DataFrame): DataFrame with engagement_ratio and niche columns
        niche_col (str): Name of the niche/category column
        percentile_high (float): Percentile threshold for High label (default 75)
        percentile_low (float): Percentile threshold for Low label (default 25)
        
    Returns:
        pd.DataFrame: DataFrame with new 'perf_label' column
        
    Notes:
        - Groups by niche before computing percentiles
        - High: engagement_ratio >= 75th percentile
        - Low: engagement_ratio <= 25th percentile
        - Middle 50% labeled as None or excluded from training
        - Returns only High/Low rows (drops middle)
    """
    pass


def add_labels_by_percentile(
    features_csv: str, 
    niche_col: str = "niche",
    output_csv: Optional[str] = None
) -> Path:
    """
    Load features CSV, add labels, and save to final_dataset.csv.
    
    Args:
        features_csv (str): Path to features_merged.csv
        niche_col (str): Name of the niche column (default "niche")
        output_csv (str): Output path (default data/final_dataset.csv)
        
    Returns:
        Path: Path to the created final_dataset.csv
        
    Steps:
        1. Load features_merged.csv
        2. Compute engagement_ratio
        3. Assign labels by percentile within each niche
        4. Filter to keep only High/Low rows
        5. Save to final_dataset.csv
        
    Example:
        >>> add_labels_by_percentile('data/processed/features_merged.csv')
        PosixPath('data/final_dataset.csv')
    """
    pass


def main():
    """
    CLI entry point for labeling.
    
    Usage:
        python -m src.labeler --in data/processed/features_merged.csv --out data/final_dataset.csv
    """
    pass


if __name__ == "__main__":
    main()

