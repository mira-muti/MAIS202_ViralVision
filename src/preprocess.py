"""
File: preprocess.py

Purpose: 
    Clean and prepare TikTok video data from multiple CSV files.
    Combines all niche CSVs, computes features, and creates a labeled dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_all_csvs(data_dir: str = "data/raw"):
    """
    Load all CSV files from the specified directory and combine them into a single DataFrame.
    
    Extracts the niche category from each filename (e.g., "music.csv" -> "music") and adds
    it as a new column. Excludes metadata_template.csv from processing.
    
    Args:
        data_dir: Path to directory containing CSV files. Defaults to "data/raw".
        
    Returns:
        pd.DataFrame: Combined dataframe with all rows from all CSV files, including a 'niche' column.
    """
    data_path = Path(data_dir)
    
    # Find all CSV files, excluding the metadata template
    csv_files = [f for f in data_path.glob("*.csv") if f.name != "metadata_template.csv"]
    
    all_dataframes = []
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        
        # Extract niche from filename (e.g., "music.csv" -> "music")
        niche_name = csv_file.stem
        df['niche'] = niche_name
        
        all_dataframes.append(df)
        print(f"Loaded {csv_file.name} with {len(df)} rows")
    
    # Concatenate all dataframes into a single dataframe
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    return combined_df


def compute_features(df):
    """
    Compute derived features from existing columns.
    
    Calculates:
    - engagement_ratio: likes divided by views (proportion of viewers who liked)
    - caption_length: character count of the title
    - hashtag_count: number of hashtags in the hashtags column (split by '#')
    
    Args:
        df: Input dataframe with columns: likes, views, title, hashtags
        
    Returns:
        pd.DataFrame: Dataframe with additional computed feature columns
    """
    df = df.copy()
    
    # Compute engagement ratio (likes/views), handling zero views
    df['engagement_ratio'] = df['likes'] / df['views'].replace(0, np.nan)
    
    # Compute caption length as character count
    df['caption_length'] = df['title'].astype(str).str.len()
    
    # Count hashtags by splitting on '#' and subtracting 1 (first empty string)
    df['hashtag_count'] = df['hashtags'].astype(str).str.split('#').str.len() - 1
    
    # Ensure non-negative hashtag count
    df['hashtag_count'] = df['hashtag_count'].clip(lower=0)
    
    return df


def clean_data(df):
    """
    Remove rows with missing or invalid engagement metrics.
    
    Filters out rows where views or likes are NaN, zero, or negative.
    Only keeps rows with valid positive engagement data.
    
    Args:
        df: Input dataframe with 'views' and 'likes' columns
        
    Returns:
        pd.DataFrame: Cleaned dataframe with only valid engagement data
    """
    initial_count = len(df)
    
    # Remove rows with missing views or likes
    df = df.dropna(subset=['views', 'likes'])
    
    # Keep only rows with positive views and likes
    df = df[(df['views'] > 0) & (df['likes'] > 0)]
    
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} rows with missing or zero views/likes")
    
    return df


def select_columns(df):
    """
    Select and retain only the specified feature columns.
    
    Args:
        df: Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe containing only the specified columns
    """
    columns_to_keep = [
        'engagement_ratio',
        'caption_length',
        'hashtag_count',
        'fft_max_freq',
        'fft_max_amp',
        'niche'
    ]
    
    # Filter to only include columns that exist in the dataframe
    available_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[available_columns]
    
    return df


def label_engagement(df):
    """
    Label engagement levels based on percentile ranking and filter middle range.
    
    Categorizes videos into High (top 30%) and Low (bottom 30%) engagement based on
    engagement_ratio percentiles. Drops the middle 40% of videos.
    
    Args:
        df: Input dataframe with 'engagement_ratio' column
        
    Returns:
        pd.DataFrame: Dataframe with 'label' column ('High' or 'Low'), middle 40% removed
    """
    # Calculate percentile thresholds
    p30 = df['engagement_ratio'].quantile(0.30)
    p70 = df['engagement_ratio'].quantile(0.70)
    
    print(f"30th percentile (cutoff for Low): {p30:.6f}")
    print(f"70th percentile (cutoff for High): {p70:.6f}")
    
    # Initialize label column
    df['label'] = None
    
    # Label top 30% as High (>= 70th percentile)
    df.loc[df['engagement_ratio'] >= p70, 'label'] = 'High'
    
    # Label bottom 30% as Low (<= 30th percentile)
    df.loc[df['engagement_ratio'] <= p30, 'label'] = 'Low'
    
    # Report distribution before filtering
    high_count = (df['label'] == 'High').sum()
    low_count = (df['label'] == 'Low').sum()
    middle_count = df['label'].isna().sum()
    
    print(f"High engagement: {high_count} rows")
    print(f"Low engagement: {low_count} rows")
    print(f"Middle (dropping): {middle_count} rows")
    
    # Remove middle 40% (rows where label is None)
    df = df.dropna(subset=['label'])
    
    return df


def main():
    """
    Execute the complete preprocessing pipeline.
    
    Orchestrates data loading, feature computation, cleaning, column selection,
    engagement labeling, and dataset export.
    """
    print("=" * 50)
    print("Starting preprocessing pipeline...")
    print("=" * 50)
    
    # Load all CSV files and add niche column
    print("\nStep 1: Loading all CSV files...")
    df = load_all_csvs("data/raw")
    print(f"Total rows loaded: {len(df)}")
    
    # Compute derived features
    print("\nStep 2-3: Computing features...")
    df = compute_features(df)
    
    # Clean invalid data
    print("\nStep 4: Cleaning data...")
    df = clean_data(df)
    print(f"Rows after cleaning: {len(df)}")
    
    # Select relevant columns
    print("\nStep 5: Selecting columns...")
    df = select_columns(df)
    
    # Label engagement levels
    print("\nStep 6: Labeling engagement...")
    df = label_engagement(df)
    print(f"Final rows: {len(df)}")
    
    # Save processed dataset
    print("\nStep 7: Saving final dataset...")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "final_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("Preprocessing complete!")
    print("=" * 50)
    print(f"\nFinal dataset summary:")
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"\nNiche distribution:")
    print(df['niche'].value_counts())


if __name__ == "__main__":
    main()
