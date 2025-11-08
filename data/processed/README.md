# Processed Data Directory

**Purpose:** Store intermediate feature extraction results.

**Owner:** Backend Pod (Person 1)

**Notes:**
- This directory contains extracted features saved as CSV files
- Each CSV may represent features from a batch of videos
- Use this for debugging and incremental processing
- Files here may be committed to Git (unless very large)

**Expected Files:**
- `audio_features.csv` - Extracted audio features
- `visual_features.csv` - Extracted visual features
- `batch_01.csv`, `batch_02.csv`, etc. - Incremental processing results

**Usage:**
These intermediate files are combined and cleaned to create `data/final_dataset.csv`

