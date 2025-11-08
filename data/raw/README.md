# Raw Data Directory

**Purpose:** Store raw video files for feature extraction.

**Owner:** Backend Pod (Person 1)

**Notes:**
- This directory is ignored by Git (videos are too large)
- Place all `.mp4`, `.avi`, `.mov`, or other video files here
- Expected format: video files with unique filenames
- Videos should be labeled/categorized for training (e.g., viral vs. not viral)
- Consider organizing in subdirectories: `viral/` and `not_viral/`

**Usage:**
```bash
# Example structure:
data/raw/
  ├── viral/
  │   ├── video001.mp4
  │   ├── video002.mp4
  │   └── ...
  └── not_viral/
      ├── video101.mp4
      ├── video102.mp4
      └── ...
```

**Collaboration:**
- Share video files via Google Drive or Dropbox if needed
- Document video sources and labeling criteria
- Ensure consistent naming convention across the team

