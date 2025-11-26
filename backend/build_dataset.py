"""
Dataset builder for ViralVision.

Downloads videos from URLs, extracts features, and builds training dataset.
Fully populates all feature columns in final_dataset.csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Optional

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from extract_features import download_video, is_local_file
from extract_features.audio_features import extract_audio_features
from extract_features.visual_features import extract_visual_features


def load_url_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV with video URLs.
    
    Expected columns:
    - url: Video URL or local path
    - title: Video title/caption
    - hashtags: Hashtag string
    - views: (optional) View count
    - likes: (optional) Like count
    - niche: (optional) Niche label (GRWM, OOTD, etc.)
    - label: (optional) Pre-labeled engagement (High=1, Low=0)
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required = ['url']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df


def compute_text_features(title: str, hashtags: str) -> Dict[str, int]:
    """Compute text-based features."""
    title = str(title) if pd.notna(title) else ''
    hashtags = str(hashtags) if pd.notna(hashtags) else ''
    
    caption_length = len(title.strip())
    
    hashtag_count = 0
    if hashtags:
        hashtag_tokens = hashtags.split()
        hashtag_count = sum(1 for token in hashtag_tokens if token.startswith("#"))
    
    return {
        'caption_length': caption_length,
        'hashtag_count': hashtag_count,
    }


def compute_engagement_ratio(views: float, likes: float) -> float:
    """Compute engagement ratio."""
    views = float(views) if pd.notna(views) else 0.0
    likes = float(likes) if pd.notna(likes) else 0.0
    
    if views > 0:
        return likes / views
    return 0.0


def extract_features_for_video(
    video_path: Path,
    title: str,
    hashtags: str,
    views: Optional[float] = None,
    likes: Optional[float] = None,
    niche: Optional[str] = None,
) -> Optional[Dict]:
    """
    Extract all features for a single video.
    
    Returns dictionary with all features, or None if extraction fails.
    Missing features are set to None.
    """
    # Initialize feature dict with None values
    features = {
        "engagement_ratio": None,
        "caption_length": None,
        "hashtag_count": None,
        "fft_max_freq": None,
        "fft_max_amp": None,
        "niche": None,
        "label": None,
        "rms_energy": None,
        "zcr": None,
        "spectral_centroid": None,
        "spectral_rolloff": None,
        "tempo": None,
        "avg_brightness": None,
        "avg_color_variance": None,
        "motion_intensity": None,
    }
    
    try:
        # Compute text features
        text_features = compute_text_features(title, hashtags)
        features['caption_length'] = text_features['caption_length']
        features['hashtag_count'] = text_features['hashtag_count']
        
        # Compute engagement ratio
        features['engagement_ratio'] = compute_engagement_ratio(views, likes)
        
        # Set niche
        features['niche'] = str(niche) if pd.notna(niche) else ''
        
        # Extract audio features
        try:
            audio_features = extract_audio_features(str(video_path), max_seconds=3.0)
            features['rms_energy'] = audio_features.get('rms_energy')
            features['zcr'] = audio_features.get('zcr')
            features['spectral_centroid'] = audio_features.get('spectral_centroid')
            features['spectral_rolloff'] = audio_features.get('spectral_rolloff')
            features['tempo'] = audio_features.get('tempo')
            features['fft_max_freq'] = audio_features.get('fft_max_freq')
            features['fft_max_amp'] = audio_features.get('fft_max_amp')
        except Exception as e:
            print(f"      ⚠ Audio extraction failed: {e}")
        
        # Extract visual features
        try:
            visual_features = extract_visual_features(str(video_path), max_seconds=3.0)
            features['avg_brightness'] = visual_features.get('avg_brightness')
            # Map color_std_dev to avg_color_variance
            features['avg_color_variance'] = visual_features.get('color_std_dev')
            features['motion_intensity'] = visual_features.get('motion_intensity')
        except Exception as e:
            print(f"      ⚠ Visual extraction failed: {e}")
        
        return features
        
    except Exception as e:
        print(f"      ✗ Error extracting features: {e}")
        return None


def label_engagement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label engagement as High (top 30%) or Low (bottom 30%), drop middle 40%.
    """
    # Remove rows with zero or missing engagement_ratio
    df = df[df['engagement_ratio'] > 0]
    df = df.dropna(subset=['engagement_ratio'])
    
    if len(df) == 0:
        raise ValueError("No valid engagement ratios found")
    
    # Compute percentiles
    high_threshold = df['engagement_ratio'].quantile(0.70)  # Top 30%
    low_threshold = df['engagement_ratio'].quantile(0.30)   # Bottom 30%
    
    # Label: 1 = High, 0 = Low
    df['label'] = np.where(df['engagement_ratio'] >= high_threshold, 1, 0)
    
    # Keep only High and Low (drop middle 40%)
    df = df[df['label'].isin([0, 1])]
    
    print(f"\nEngagement labeling:")
    print(f"  High threshold: {high_threshold:.6f}")
    print(f"  Low threshold: {low_threshold:.6f}")
    print(f"  High (1): {(df['label'] == 1).sum()}")
    print(f"  Low (0): {(df['label'] == 0).sum()}")
    
    return df


def build_dataset(
    csv_path: Path,
    output_path: Path,
    videos_dir: Path,
    max_rows: Optional[int] = None,
    use_existing_labels: bool = False,
) -> None:
    """
    Main dataset building pipeline.
    
    Fully populates all feature columns and saves to final_dataset.csv.
    """
    print("=" * 60)
    print("ViralVision Dataset Builder")
    print("=" * 60)
    
    # Load CSV
    print(f"\n[1] Loading CSV: {csv_path}")
    df = load_url_csv(csv_path)
    print(f"  ✓ Loaded {len(df)} rows")
    
    # Filter to GRWM/OOTD if niche column exists
    if 'niche' in df.columns:
        allowed = {'GRWM', 'grwm', 'OOTD', 'ootd'}
        before = len(df)
        df = df[df['niche'].astype(str).str.upper().isin([n.upper() for n in allowed])]
        after = len(df)
        print(f"  Filtered to GRWM/OOTD: {before} -> {after} rows")
    
    # Limit rows for testing
    if max_rows:
        df = df.head(max_rows)
        print(f"  Limited to {max_rows} rows for processing")
    
    # Process videos
    print(f"\n[2] Processing {len(df)} videos...")
    processed_rows = []
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    # Track success/failure counts
    download_success = 0
    download_failed = 0
    feature_extraction_failed = 0
    
    for idx, row in df.iterrows():
        url = row.get('url', '')
        title = row.get('title', '')
        hashtags = row.get('hashtags', '')
        views = row.get('views', None)
        likes = row.get('likes', None)
        niche = row.get('niche', 'GRWM')
        
        if not url or pd.isna(url):
            print(f"  [{idx+1}/{len(df)}] Skipping row {idx}: no URL")
            continue
        
        print(f"  [{idx+1}/{len(df)}] Downloading:")
        print(f"      URL: {url[:80]}")
        
        video_path = None
        try:
            # Download or get local file
            video_path = download_video(url, videos_dir)
            
            # Check if download failed
            if video_path is None:
                download_failed += 1
                print(f"      → FAILED: Video download failed — skipping")
                continue
            
            # Download succeeded
            download_success += 1
            print(f"      → Download OK: {video_path.name}")
            
            # Extract features
            print(f"      Extracting features...")
            features = extract_features_for_video(
                video_path, title, hashtags, views, likes, niche
            )
            
            if features:
                # Add label if provided in CSV
                if use_existing_labels and 'label' in row and pd.notna(row['label']):
                    features['label'] = int(row['label'])
                
                processed_rows.append(features)
                print(f"      → ✓ Success: Features extracted")
            else:
                feature_extraction_failed += 1
                print(f"      → ✗ Feature extraction failed")
                
        except Exception as e:
            download_failed += 1
            print(f"      → ✗ Error: {e}")
        finally:
            pass
    
    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Download Summary:")
    print(f"  Downloaded successfully: {download_success}/{len(df)}")
    print(f"  Download failed: {download_failed}")
    print(f"  Feature extraction failed: {feature_extraction_failed}")
    print(f"  Successfully processed: {len(processed_rows)}/{len(df)}")
    print(f"{'=' * 60}")
    
    if not processed_rows:
        raise ValueError("No videos successfully processed")
    
    print(f"\n[3] Building final dataset...")
    final_df = pd.DataFrame(processed_rows)
    
    # Label engagement if not already labeled
    if 'label' not in final_df.columns or not use_existing_labels:
        final_df = label_engagement(final_df)
    
    # Define exact column order
    columns = [
        "engagement_ratio",
        "caption_length",
        "hashtag_count",
        "fft_max_freq",
        "fft_max_amp",
        "niche",
        "label",
        "rms_energy",
        "zcr",
        "spectral_centroid",
        "spectral_rolloff",
        "tempo",
        "avg_brightness",
        "avg_color_variance",
        "motion_intensity",
    ]
    
    # Ensure all columns exist (fill missing with None)
    for col in columns:
        if col not in final_df.columns:
            final_df[col] = None
    
    # Reorder columns
    final_df = final_df[columns]
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    
    # Print detailed summary
    print(f"\n[4] Saved dataset: {output_path}")
    print(f"  Total rows: {len(final_df)}")
    print(f"  Total columns: {len(final_df.columns)}")
    
    # Feature completeness analysis
    print(f"\n  Feature Completeness:")
    for col in columns:
        if col in ['niche', 'label']:
            continue  # Skip categorical columns
        non_null = final_df[col].notna().sum()
        pct = (non_null / len(final_df)) * 100 if len(final_df) > 0 else 0
        print(f"    {col:20s}: {non_null:4d}/{len(final_df):4d} ({pct:5.1f}%)")
    
    if 'niche' in final_df.columns:
        print(f"\n  Niche distribution:")
        print(f"    {final_df['niche'].value_counts().to_dict()}")
    
    print(f"\n{'=' * 60}")
    print("Dataset building complete!")
    print(f"{'=' * 60}")


def main():
    """Main entrypoint."""
    project_root = Path(__file__).parent.parent
    
    # Default paths - save to final_dataset.csv
    csv_path = project_root / "data" / "raw" / "GRWM.csv"  # Can be changed
    output_path = project_root / "data" / "processed" / "final_dataset.csv"
    videos_dir = project_root / "data" / "videos"
    
    # Optional: limit rows for testing
    max_rows = None  # Set to e.g., 10 for quick testing
    
    build_dataset(csv_path, output_path, videos_dir, max_rows=max_rows)


if __name__ == "__main__":
    main()
