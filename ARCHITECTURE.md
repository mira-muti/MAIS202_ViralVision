# 🏗️ MAIS202_ViralVision Architecture

## High-Level Data Pipeline

```
[ TikTok Scraper ] → metadata.csv
         │
         ├─▶ [Video Downloader] → videos/*.mp4
         │
         ├─▶ [Audio Extractor (ffmpeg)] → audio/*.wav (first 5s)
         │
         └─▶ [Frame Sampler (ffmpeg)] → frames/<video_id>/frame_001.jpg ... frame_005.jpg
                                         (1 fps for first 5s)
              │
              ▼
   [Feature Extractor (librosa + opencv)]
     ├─ audio_features.csv
     ├─ visual_features.csv
     └─ features_merged.csv  (one row per video)
              │
              ▼
     [Labeler] → final_dataset.csv (adds High/Low by percentile)
              │
              ▼
     [Model Trainer] → models/model.pkl  (+ metrics.json, report.png)
              │
              ▼
     [Inference API] → predict_from_video(video.mp4) → {score, label, top_features}
              │
              ▼
     [Streamlit App] → upload → predict → show score + insights
```

---

## Repository Layout

```
MAIS202_ViralVision/
├── data/
│   ├── raw/                 # metadata.csv from scraper + temp files
│   ├── processed/           # audio_features.csv, visual_features.csv
│   └── final_dataset.csv    # training table with labels
│
├── media/
│   ├── videos/              # *.mp4 (not committed)
│   ├── audio/               # *.wav (not committed)
│   └── frames/              # frames/<video_id>/*.jpg (not committed)
│
├── src/
│   ├── audio_features.py    # librosa feature functions
│   ├── visual_features.py   # opencv feature functions
│   ├── preprocess.py        # feature extraction glue / batch runner
│   ├── labeler.py           # percentile labels High/Low
│   ├── train_model.py       # fit RF/NB, save model.pkl
│   ├── predict.py           # predict_from_video()
│   └── utils.py             # helper functions
│
├── models/
│   └── model.pkl
│
├── app/
│   └── streamlit_app.py
│
├── notebooks/
│   └── EDA.ipynb
│
├── scripts/                 # tiny CLIs (bash/py) to run steps
│   ├── 01_download_videos.sh
│   ├── 02_extract_audio.sh
│   ├── 03_sample_frames.sh
│   └── 04_build_dataset.py
│
├── requirements.txt
├── README.md
├── TEAM_GUIDE.md
├── ARCHITECTURE.md          # This file
└── .gitignore
```

---

## Data Contracts

### 1. data/raw/metadata.csv (from scraper)

One row per video.

| Column | Type | Example |
|--------|------|---------|
| video_id | str | 7239401123 |
| creator_handle | str | @fitcoach |
| video_url | str | https://www.tiktok.com/@... |
| likes | int | 125034 |
| comments | int | 1032 |
| shares | int | 756 |
| views | int | 980000 |
| caption | str | "3 exercises to fix your squat" |
| niche | str (categorical) | fitness |
| downloaded_path | str | media/videos/7239401123.mp4 |

### 2. data/processed/audio_features.csv

| Column | Type | Description |
|--------|------|-------------|
| video_id | str | Unique identifier |
| rms | float | Root mean square energy (loudness) |
| tempo_bpm | float | Estimated tempo in beats per minute |
| spectral_centroid | float | Mean spectral centroid (brightness) |
| zcr | float | Zero-crossing rate |
| mfcc1, mfcc2, ..., mfcc5 | float | Mean of first 5 MFCCs |

All numeric; computed over first 5 seconds of audio.

### 3. data/processed/visual_features.csv

| Column | Type | Description |
|--------|------|-------------|
| video_id | str | Unique identifier |
| avg_brightness | float | Average brightness (0-255) |
| contrast | float | Contrast measure (std of intensities) |
| color_var | float | Color variance across frames |
| motion_score | float | Motion between consecutive frames |
| faces | int | Average face count (optional) |

Computed from 5 frames sampled at 1 fps over first 5 seconds.

### 4. data/processed/features_merged.csv

Join by `video_id` + keep engagement columns.
Contains all columns from metadata + audio features + visual features.

### 5. data/final_dataset.csv

Adds:
- `engagement_ratio = likes / views`
- `perf_label ∈ {High, Low}` using niche-wise percentiles
  - Top 25% = High
  - Bottom 25% = Low
  - Middle 50% excluded for crisp classification

---

## Module Responsibilities

### src/audio_features.py (Person 1)

**Functions:**
- `load_audio_first5(audio_path, sr)` → np.ndarray
- `compute_rms(y)` → float
- `estimate_tempo(y, sr)` → float
- `spectral_centroid(y, sr)` → float
- `zero_crossing_rate(y)` → float
- `mfcc_stats(y, sr, n)` → Dict[str, float]
- `extract_audio_features(audio_path)` → Dict[str, float]

**Purpose:** Extract audio features using librosa

### src/visual_features.py (Person 1)

**Functions:**
- `load_frames(folder)` → List[np.ndarray]
- `avg_brightness(frames)` → float
- `contrast(frames)` → float
- `color_variance(frames)` → float
- `motion_score(frames)` → float
- `face_count(frames)` → int (optional)
- `extract_visual_features(frames_folder)` → Dict[str, float]

**Purpose:** Extract visual features using OpenCV

### src/preprocess.py (Person 1)

**Functions:**
- `extract_all_audio_features(meta_csv, audio_dir)` → Path
- `extract_all_visual_features(meta_csv, frames_root)` → Path
- `merge_features(meta_csv, audio_csv, visual_csv)` → Path

**Purpose:** Glue code for batch processing

### src/labeler.py (Person 2)

**Functions:**
- `compute_engagement_ratio(df)` → DataFrame
- `assign_labels_by_percentile(df, niche_col)` → DataFrame
- `add_labels_by_percentile(features_csv, niche_col)` → Path

**Purpose:** Add High/Low labels based on engagement percentiles

### src/train_model.py (Person 2)

**Functions:**
- `load_dataset(csv_path)` → (X, y)
- `train_random_forest(X_train, y_train)` → (model, metrics)
- `train_naive_bayes(X_train, y_train)` → (model, metrics)
- `evaluate_model(model, X_test, y_test)` → Dict
- `save_model(model, out_path)` → None

**Purpose:** Train and save ML models

### src/predict.py (Person 2 lead)

**Functions:**
- `extract_features_from_video(video_path)` → DataFrame
- `load_trained_model(model_path)` → model
- `predict_from_video(video_path, model_path)` → Dict[str, Any]

**Returns from predict_from_video:**
```python
{
  "score": 78,  # 0-100
  "label": "High",  # or "Low"
  "top_features": [
    {"name": "tempo_bpm", "direction": "+", "contrib": 0.25},
    {"name": "motion_score", "direction": "+", "contrib": 0.18},
    ...
  ]
}
```

**Purpose:** Inference on new videos

---

## CLI / Batch Steps

### 0. Environment Setup

```bash
conda create -n viralvision python=3.10 -y
conda activate viralvision
pip install -r requirements.txt
```

### 1. Download Videos

```bash
bash scripts/01_download_videos.sh
```
**Output:** `data/raw/metadata.csv` and files in `media/videos/*.mp4`

### 2. Extract Audio (first 5 seconds, mono 44.1kHz)

```bash
bash scripts/02_extract_audio.sh
```
**Output:** `media/audio/*.wav`

### 3. Sample Frames (1 fps for first 5s)

```bash
bash scripts/03_sample_frames.sh
```
**Output:** `media/frames/<video_id>/*.jpg`

### 4. Extract Features & Merge

```bash
python scripts/04_build_dataset.py \
  --meta data/raw/metadata.csv \
  --audio_dir media/audio \
  --frames_dir media/frames \
  --out_dir data/processed
```
**Output:** `audio_features.csv`, `visual_features.csv`, `features_merged.csv`

### 5. Label and Finalize Dataset

```bash
python -m src.labeler \
  --in data/processed/features_merged.csv \
  --out data/final_dataset.csv
```
**Output:** `data/final_dataset.csv` with High/Low labels

### 6. Train Models

```bash
python -m src.train_model \
  --data data/final_dataset.csv \
  --out models/model.pkl \
  --report data/processed/metrics.json
```
**Output:** `models/model.pkl`, `metrics.json`

### 7. Inference Sanity Check

```bash
python -m src.predict \
  --video media/videos/TEST.mp4 \
  --model models/model.pkl
```
**Output:** JSON with score, label, top_features

### 8. Run App

```bash
streamlit run app/streamlit_app.py
```

---

## Division of Labor

### Backend Pod (Pair) — Branch: `data-model`

**Person 1 (Feature Extraction Lead):**
- `scripts/*.sh` (bash scripts)
- `src/audio_features.py`
- `src/visual_features.py`
- `src/preprocess.py`

**Person 2 (Model Lead):**
- `src/labeler.py`
- `src/train_model.py`
- `src/predict.py`
- `notebooks/EDA.ipynb`
- `models/`

### Frontend (Beginner) — Branch: `webapp`

- `app/streamlit_app.py` only
- Receives `models/model.pkl` and imports:
  - `from src.predict import predict_from_video`

---

## Labeling Logic

```python
engagement_ratio = likes / max(views, 1)

# Within each niche:
perf_label = "High" if engagement_ratio >= 75th percentile
perf_label = "Low"  if engagement_ratio <= 25th percentile
# (ignore middle 50% for crisp classes)
```

---

## Minimal Feature Set (2-Week Friendly)

**Audio (first 5s):**
- rms, tempo_bpm, spectral_centroid, zcr, mfcc1..mfcc5_mean

**Visual (5 frames):**
- avg_brightness, contrast, color_var, motion_score, faces (optional)

**Total:** ~12–16 features — perfect for Random Forest / Naive Bayes on small data.

---

## Quality Checks

1. **Data completeness:** % videos with 5 frames extracted; % with audio present
2. **Feature sanity:** ranges (rms in [0,1]? tempo 60–180?), spot outliers
3. **Baseline vs model:** majority accuracy vs RF/NB accuracy/F1
4. **Hold-out test:** report metrics on 30% split

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Scraper breaks / rate limits | Cache metadata.csv; start with ~60 videos |
| No audio in some clips | Skip or impute; keep a `has_audio` flag |
| ffmpeg not installed on teammate laptop | Add install instructions in README |
| Git conflicts in pod | File ownership + small commits + one shared branch |

---

## Feature Descriptions (for Streamlit display)

| Feature | User-Friendly Name | Description |
|---------|-------------------|-------------|
| rms | Audio Energy | How loud the video is |
| tempo_bpm | Beat Tempo | Speed of the music/audio |
| spectral_centroid | Audio Brightness | High vs low frequency sounds |
| zcr | Audio Texture | Noisiness of the audio |
| mfcc1-5 | Audio Timbre | Unique "voice" of the audio |
| avg_brightness | Visual Brightness | How bright the video appears |
| contrast | Visual Contrast | Difference between light and dark |
| color_var | Color Variety | How colorful the video is |
| motion_score | Movement | How much action/motion there is |
| faces | Face Count | Number of people visible |

---

**Last Updated:** 2025-11-08

