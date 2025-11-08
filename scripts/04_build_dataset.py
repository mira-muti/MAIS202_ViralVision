#!/usr/bin/env python3
"""
Script: 04_build_dataset.py
Owner: Backend Pod – Person 1
Purpose: Extract features and merge into final dataset

This script:
1. Extracts audio features from all WAV files
2. Extracts visual features from all frame directories
3. Merges with metadata
4. Saves intermediate CSVs

Usage:
    python scripts/04_build_dataset.py \
        --meta data/raw/metadata.csv \
        --audio_dir media/audio \
        --frames_dir media/frames \
        --out_dir data/processed
"""

import argparse
import pandas as pd
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# from audio_features import extract_audio_features
# from visual_features import extract_visual_features


def extract_all_audio_features(meta_csv: str, audio_dir: str, out_dir: str) -> Path:
    """
    Extract audio features for all videos and save to CSV.
    
    Args:
        meta_csv: Path to metadata.csv
        audio_dir: Directory containing .wav files
        out_dir: Output directory for audio_features.csv
        
    Returns:
        Path to audio_features.csv
    """
    print("\n" + "="*50)
    print("Extracting Audio Features")
    print("="*50)
    
    # TODO: Implement audio feature extraction
    # 1. Read metadata.csv
    # 2. For each video_id, extract features from audio_dir/{video_id}.wav
    # 3. Collect all features into a DataFrame
    # 4. Save to data/processed/audio_features.csv
    
    print("⚠️  TODO: Implement audio feature extraction")
    print("   Expected output: data/processed/audio_features.csv")
    
    out_path = Path(out_dir) / "audio_features.csv"
    return out_path


def extract_all_visual_features(meta_csv: str, frames_dir: str, out_dir: str) -> Path:
    """
    Extract visual features for all videos and save to CSV.
    
    Args:
        meta_csv: Path to metadata.csv
        frames_dir: Root directory containing frame subdirectories
        out_dir: Output directory for visual_features.csv
        
    Returns:
        Path to visual_features.csv
    """
    print("\n" + "="*50)
    print("Extracting Visual Features")
    print("="*50)
    
    # TODO: Implement visual feature extraction
    # 1. Read metadata.csv
    # 2. For each video_id, extract features from frames_dir/{video_id}/
    # 3. Collect all features into a DataFrame
    # 4. Save to data/processed/visual_features.csv
    
    print("⚠️  TODO: Implement visual feature extraction")
    print("   Expected output: data/processed/visual_features.csv")
    
    out_path = Path(out_dir) / "visual_features.csv"
    return out_path


def merge_features(meta_csv: str, audio_csv: str, visual_csv: str, out_dir: str) -> Path:
    """
    Merge metadata, audio features, and visual features into one CSV.
    
    Args:
        meta_csv: Path to metadata.csv
        audio_csv: Path to audio_features.csv
        visual_csv: Path to visual_features.csv
        out_dir: Output directory for features_merged.csv
        
    Returns:
        Path to features_merged.csv
    """
    print("\n" + "="*50)
    print("Merging Features")
    print("="*50)
    
    # TODO: Implement feature merging
    # 1. Load metadata.csv
    # 2. Load audio_features.csv
    # 3. Load visual_features.csv
    # 4. Merge all on video_id
    # 5. Keep engagement columns (likes, views, etc.)
    # 6. Save to data/processed/features_merged.csv
    
    print("⚠️  TODO: Implement feature merging")
    print("   Expected output: data/processed/features_merged.csv")
    
    out_path = Path(out_dir) / "features_merged.csv"
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build feature dataset")
    parser.add_argument("--meta", required=True, help="Path to metadata.csv")
    parser.add_argument("--audio_dir", required=True, help="Directory with audio files")
    parser.add_argument("--frames_dir", required=True, help="Directory with frame subdirs")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract audio features
    audio_csv = extract_all_audio_features(args.meta, args.audio_dir, args.out_dir)
    
    # Extract visual features
    visual_csv = extract_all_visual_features(args.meta, args.frames_dir, args.out_dir)
    
    # Merge all features
    merged_csv = merge_features(args.meta, str(audio_csv), str(visual_csv), args.out_dir)
    
    print("\n" + "="*50)
    print("✅ Dataset Building Complete")
    print("="*50)
    print(f"Output: {merged_csv}")
    print("\nNext step: Run labeling script")
    print("  python -m src.labeler --in data/processed/features_merged.csv --out data/final_dataset.csv")


if __name__ == "__main__":
    main()

