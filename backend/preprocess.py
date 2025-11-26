"""
Preprocessing pipeline for ViralVision dataset.

This script processes raw TikTok CSV files and extracts audio, visual,
and text features to create the final training dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import sys
import os

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from extract_audio_features import extract_fft_features, extract_audio_feature_dict
from extract_visual_features import extract_visual_features

# Paths
project_root = Path(__file__).parent.parent
RAW_DATA_PATHS = [
    project_root / "data" / "raw",
    project_root / "backend" / "data" / "raw",
    project_root / "data",
]
OUTPUT_DIR = project_root / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "final_dataset.csv"

# Required columns in raw CSV
REQUIRED_COLUMNS = ['url', 'title', 'hashtags', 'views', 'likes', 'fft_max_freq', 'fft_max_amp']


def load_raw_csvs() -> pd.DataFrame:
    """
    Load and combine all raw CSV files from multiple locations.
    
    Returns:
        Combined DataFrame with all raw data
    """
    all_dataframes = []
    
    for raw_path in RAW_DATA_PATHS:
        if not raw_path.exists():
            continue
        
        csv_files = list(raw_path.glob("*.csv"))
        print(f"Searching in {raw_path}: found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Check if CSV has required columns
                missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
                if missing_cols:
                    print(f"  Skipping {csv_file.name}: missing columns {missing_cols}")
                    continue
                
                # Add source filename for niche inference
                df['_source_file'] = csv_file.stem
                all_dataframes.append(df)
                print(f"  ✓ Loaded {csv_file.name}: {len(df)} rows")
                
            except Exception as e:
                print(f"  ✗ Error loading {csv_file.name}: {str(e)}")
                continue
    
    if not all_dataframes:
        raise ValueError("No valid CSV files found in any of the search paths")
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\nTotal rows loaded: {len(combined_df)}")
    
    return combined_df


def extract_features_for_row(row: pd.Series, video_base_path: Optional[Path] = None) -> Dict:
    """
    Extract all features for a single row.
    
    Args:
        row: DataFrame row with raw data
        video_base_path: Optional base path for video files
        
    Returns:
        Dictionary with all extracted features
    """
    features = {}
    
    # 1. Text features
    title = str(row.get('title', '')) if pd.notna(row.get('title')) else ''
    hashtags = str(row.get('hashtags', '')) if pd.notna(row.get('hashtags')) else ''
    
    features['caption_length'] = len(title.strip())
    
    # Count hashtags (tokens starting with "#")
    hashtag_count = 0
    if hashtags:
        hashtag_tokens = hashtags.split()
        hashtag_count = sum(1 for token in hashtag_tokens if token.startswith("#"))
    features['hashtag_count'] = hashtag_count
    
    # 2. Audio features (from existing CSV or extract from video)
    # First, try to use existing fft_max_freq and fft_max_amp from CSV
    fft_max_freq = row.get('fft_max_freq')
    fft_max_amp = row.get('fft_max_amp')
    
    # If not available or invalid, try to extract from video
    if pd.isna(fft_max_freq) or pd.isna(fft_max_amp):
        video_path = None
        
        # Try to find video file from URL or other fields
        url = str(row.get('url', '')) if pd.notna(row.get('url')) else ''
        
        # If we have a video_base_path, try to construct path
        if video_base_path and url:
            # Try common video filename patterns
            video_id = url.split('/')[-1] if '/' in url else None
            if video_id:
                for ext in ['.mp4', '.mov', '.avi']:
                    potential_path = video_base_path / f"{video_id}{ext}"
                    if potential_path.exists():
                        video_path = potential_path
                        break
        
        # Try to extract audio features if video found
        if video_path and video_path.exists():
            try:
                fft_max_freq, fft_max_amp = extract_fft_features(str(video_path), seconds=5)
                audio_features = extract_audio_feature_dict(str(video_path), seconds=5)
                
                features['fft_max_freq'] = fft_max_freq
                features['fft_max_amp'] = fft_max_amp
                features['rms_energy'] = audio_features.get('rms_energy', np.nan)
                features['zcr'] = audio_features.get('zcr', np.nan)
                features['spectral_centroid'] = audio_features.get('spectral_centroid', np.nan)
                features['spectral_rolloff'] = audio_features.get('spectral_rolloff', np.nan)
                features['tempo'] = audio_features.get('tempo', np.nan)
            except Exception as e:
                print(f"    Warning: Could not extract audio from {video_path}: {str(e)}")
                features['fft_max_freq'] = np.nan
                features['fft_max_amp'] = np.nan
                features['rms_energy'] = np.nan
                features['zcr'] = np.nan
                features['spectral_centroid'] = np.nan
                features['spectral_rolloff'] = np.nan
                features['tempo'] = np.nan
        else:
            # Use existing values or NaN
            features['fft_max_freq'] = fft_max_freq if not pd.isna(fft_max_freq) else np.nan
            features['fft_max_amp'] = fft_max_amp if not pd.isna(fft_max_amp) else np.nan
            features['rms_energy'] = np.nan
            features['zcr'] = np.nan
            features['spectral_centroid'] = np.nan
            features['spectral_rolloff'] = np.nan
            features['tempo'] = np.nan
    else:
        # Use existing values from CSV
        features['fft_max_freq'] = float(fft_max_freq)
        features['fft_max_amp'] = float(fft_max_amp)
        
        # Try to extract extended audio features if video available
        video_path = None
        url = str(row.get('url', '')) if pd.notna(row.get('url')) else ''
        if video_base_path and url:
            video_id = url.split('/')[-1] if '/' in url else None
            if video_id:
                for ext in ['.mp4', '.mov', '.avi']:
                    potential_path = video_base_path / f"{video_id}{ext}"
                    if potential_path.exists():
                        video_path = potential_path
                        break
        
        if video_path and video_path.exists():
            try:
                audio_features = extract_audio_feature_dict(str(video_path), seconds=5)
                features['rms_energy'] = audio_features.get('rms_energy', np.nan)
                features['zcr'] = audio_features.get('zcr', np.nan)
                features['spectral_centroid'] = audio_features.get('spectral_centroid', np.nan)
                features['spectral_rolloff'] = audio_features.get('spectral_rolloff', np.nan)
                features['tempo'] = audio_features.get('tempo', np.nan)
            except Exception:
                features['rms_energy'] = np.nan
                features['zcr'] = np.nan
                features['spectral_centroid'] = np.nan
                features['spectral_rolloff'] = np.nan
                features['tempo'] = np.nan
        else:
            features['rms_energy'] = np.nan
            features['zcr'] = np.nan
            features['spectral_centroid'] = np.nan
            features['spectral_rolloff'] = np.nan
            features['tempo'] = np.nan
    
    # 3. Visual features (extract from video if available)
    video_path = None
    url = str(row.get('url', '')) if pd.notna(row.get('url')) else ''
    if video_base_path and url:
        video_id = url.split('/')[-1] if '/' in url else None
        if video_id:
            for ext in ['.mp4', '.mov', '.avi']:
                potential_path = video_base_path / f"{video_id}{ext}"
                if potential_path.exists():
                    video_path = potential_path
                    break
    
    if video_path and video_path.exists():
        try:
            visual_features = extract_visual_features(str(video_path), max_seconds=5, frame_sample_rate=10)
            features['avg_brightness'] = visual_features.get('avg_brightness', np.nan)
            features['avg_color_variance'] = visual_features.get('avg_color_variance', np.nan)
            features['motion_intensity'] = visual_features.get('motion_intensity', np.nan)
        except Exception:
            features['avg_brightness'] = np.nan
            features['avg_color_variance'] = np.nan
            features['motion_intensity'] = np.nan
    else:
        features['avg_brightness'] = np.nan
        features['avg_color_variance'] = np.nan
        features['motion_intensity'] = np.nan
    
    # 4. Engagement ratio
    views = row.get('views', 0)
    likes = row.get('likes', 0)
    
    # Convert to numeric if string
    try:
        views = float(views) if pd.notna(views) else 0
        likes = float(likes) if pd.notna(likes) else 0
    except (ValueError, TypeError):
        views = 0
        likes = 0
    
    if views > 0:
        features['engagement_ratio'] = likes / views
    else:
        features['engagement_ratio'] = np.nan
    
    # 5. Niche (from source filename)
    source_file = row.get('_source_file', 'unknown')
    if pd.notna(source_file):
        niche = str(source_file).lower()
        # Clean up common patterns
        if 'grwm' in niche:
            niche = 'GRWM'
        elif 'music' in niche:
            niche = 'music'
        elif 'ootd' in niche:
            niche = 'OOTD'
        elif 'food' in niche:
            niche = 'food'
        elif 'dance' in niche:
            niche = 'dance'
        elif 'diy' in niche or 'project' in niche:
            niche = 'DIYProjects'
        elif 'summer' in niche:
            niche = 'summervibes'
        elif 'fyp' in niche:
            niche = 'fyp'
        else:
            niche = 'unknown'
    else:
        niche = 'unknown'
    
    features['niche'] = niche
    
    return features


def label_engagement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label engagement using percentile thresholds.
    
    Args:
        df: DataFrame with engagement_ratio
        
    Returns:
        DataFrame with label column added
    """
    # Remove rows with invalid engagement_ratio
    df = df[df['engagement_ratio'].notna() & (df['engagement_ratio'] >= 0)]
    
    if len(df) == 0:
        raise ValueError("No valid engagement ratios found")
    
    # Compute percentiles
    p30 = df['engagement_ratio'].quantile(0.30)
    p70 = df['engagement_ratio'].quantile(0.70)
    
    print(f"\nEngagement ratio percentiles:")
    print(f"  30th percentile: {p30:.6f}")
    print(f"  70th percentile: {p70:.6f}")
    
    # Label rows
    df['label'] = np.nan
    df.loc[df['engagement_ratio'] >= p70, 'label'] = 1  # High
    df.loc[df['engagement_ratio'] <= p30, 'label'] = 0  # Low
    
    # Drop middle 40%
    initial_count = len(df)
    df = df[df['label'].notna()]
    dropped_count = initial_count - len(df)
    
    print(f"\nLabeling results:")
    print(f"  High (>= p70): {(df['label'] == 1).sum()}")
    print(f"  Low (<= p30): {(df['label'] == 0).sum()}")
    print(f"  Dropped (middle 40%): {dropped_count}")
    
    return df


def build_final_dataset():
    """
    Main preprocessing pipeline.
    """
    print("="*60)
    print("ViralVision Preprocessing Pipeline")
    print("="*60)
    
    # 1. Load raw CSVs
    print("\n[1] Loading raw CSV files...")
    df = load_raw_csvs()
    
    # 2. Extract features for each row
    print(f"\n[2] Extracting features for {len(df)} rows...")
    print("    (This may take a while if videos need to be processed)")
    
    # Optional: video base path (if videos are stored locally)
    video_base_path = None
    potential_video_dirs = [
        project_root / "media" / "videos",
        project_root / "data" / "videos",
        project_root / "videos",
    ]
    for vdir in potential_video_dirs:
        if vdir.exists():
            video_base_path = vdir
            print(f"    Found video directory: {video_base_path}")
            break
    
    feature_rows = []
    for idx, row in df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"    Processing row {idx + 1}/{len(df)}...")
        
        try:
            features = extract_features_for_row(row, video_base_path)
            feature_rows.append(features)
        except Exception as e:
            print(f"    Warning: Error processing row {idx + 1}: {str(e)}")
            continue
    
    # Create feature DataFrame
    features_df = pd.DataFrame(feature_rows)
    
    # 3. Remove rows with invalid engagement_ratio
    print(f"\n[3] Filtering invalid rows...")
    initial_count = len(features_df)
    features_df = features_df[features_df['engagement_ratio'].notna() & (features_df['engagement_ratio'] >= 0)]
    filtered_count = initial_count - len(features_df)
    print(f"    Removed {filtered_count} rows with invalid engagement_ratio")
    
    # 4. Label engagement
    print(f"\n[4] Labeling engagement...")
    features_df = label_engagement(features_df)
    
    # 5. Select and order columns for output
    print(f"\n[5] Preparing final dataset...")
    
    # Required columns for model
    required_cols = [
        'engagement_ratio',
        'caption_length',
        'hashtag_count',
        'fft_max_freq',
        'fft_max_amp',
        'niche',
        'label'
    ]
    
    # Extra feature columns
    extra_cols = [
        'rms_energy',
        'zcr',
        'spectral_centroid',
        'spectral_rolloff',
        'tempo',
        'avg_brightness',
        'avg_color_variance',
        'motion_intensity'
    ]
    
    # Combine all columns
    all_cols = required_cols + extra_cols
    
    # Select only columns that exist
    output_cols = [col for col in all_cols if col in features_df.columns]
    final_df = features_df[output_cols].copy()
    
    # Ensure required columns exist (fill with NaN if missing)
    for col in required_cols:
        if col not in final_df.columns:
            final_df[col] = np.nan
    
    # Reorder columns
    final_df = final_df[output_cols]
    
    # 6. Save to CSV
    print(f"\n[6] Saving final dataset...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n{'='*60}")
    print(f"Preprocessing Complete!")
    print(f"{'='*60}")
    print(f"Final dataset: {OUTPUT_FILE}")
    print(f"Rows: {len(final_df)}")
    print(f"Columns: {len(final_df.columns)}")
    print(f"\nColumn summary:")
    for col in final_df.columns:
        non_null = final_df[col].notna().sum()
        print(f"  {col}: {non_null}/{len(final_df)} non-null")


if __name__ == "__main__":
    build_final_dataset()

