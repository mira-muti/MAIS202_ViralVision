# 🎬 ViralVision - Video Virality Predictor

## 📖 Project Overview

**ViralVision** is a machine learning application that predicts whether a video will go viral based on audio and visual features. Using Random Forest and Naive Bayes classifiers, the system analyzes video content and provides predictions with confidence scores through an intuitive Streamlit web interface.

This is a 2-week team project for MAIS202, designed to demonstrate end-to-end ML development: data preprocessing, model training, and deployment.

---

## 👥 Team Roles & Branches

### Backend Pod (2 Coders)
**Branch:** `data-model`

**Division of Labor:**
- **Person 1:** Feature extraction (`src/preprocess.py`)
  - Extract audio features (tempo, spectral features, MFCCs)
  - Extract visual features (brightness, color, motion)
  - Build final feature dataset CSV

- **Person 2:** Model training & inference (`src/train_model.py`, `src/predict.py`)
  - Train Random Forest and Naive Bayes models
  - Evaluate model performance
  - Implement prediction functions for Streamlit app

**Coordination Rules:**
- Both push/pull from `data-model` branch
- Never edit the same file simultaneously
- Agree on feature dictionary keys and CSV column names early
- Use detailed comments and docstrings for shared functions
- Share model via `models/model.pkl` (or Google Drive if too large)

### Frontend Developer (Beginner)
**Branch:** `webapp`

**Responsibilities:**
- Build Streamlit user interface (`app/streamlit_app.py`)
- Upload videos, display predictions, show visualizations
- Uses the trained model and prediction functions from Backend Pod

**Coordination Rules:**
- Works only in `app/streamlit_app.py`
- Imports functions from `src/predict.py` (provided by Backend Pod)
- Asks backend team for clarification on function signatures if needed

---

## 📂 Project Structure

```
MAIS202_ViralVision/
│
├── data/
│   ├── raw/                  # Raw video files (NOT committed to Git)
│   ├── processed/            # Extracted feature CSVs
│   └── final_dataset.csv     # Combined cleaned dataset for training
│
├── src/
│   ├── preprocess.py         # Feature extraction → Backend Pod (Person 1)
│   ├── train_model.py        # Model training → Backend Pod (Person 2)
│   ├── predict.py            # Inference functions → Backend Pod (shared)
│   └── utils.py              # Helper functions (paths, normalization, etc.)
│
├── models/
│   └── model.pkl             # Saved trained model (binary)
│
├── app/
│   └── streamlit_app.py      # Streamlit frontend → Frontend Dev
│
├── notebooks/
│   ├── EDA.ipynb             # Exploratory Data Analysis
│   └── feature_test.ipynb    # Quick tests for feature extraction
│
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore                # Git ignore rules
└── LICENSE                   # (Optional)
```

---

## 🚀 Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd MAIS202_ViralVision
```

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Branch for Your Role

**Backend Pod:**
```bash
git checkout -b data-model
```

**Frontend Dev:**
```bash
git checkout -b webapp
```

### 5. Add Raw Video Data
Place your raw video files in `data/raw/`. This directory is ignored by Git (videos are too large).

---

## 🧠 How to Train the Model

### Step 1: Extract Features (Person 1)
```bash
python src/preprocess.py
```

This will:
- Process all videos in `data/raw/`
- Extract audio and visual features
- Save the final dataset to `data/final_dataset.csv`

### Step 2: Train the Model (Person 2)
```bash
python src/train_model.py
```

This will:
- Load `data/final_dataset.csv`
- Train Random Forest and Naive Bayes models
- Evaluate and compare models
- Save the best model to `models/model.pkl`

### Step 3: Test Predictions
```bash
python src/predict.py
```

Test inference on a sample video to verify the model works correctly.

---

## 🎨 How to Run the Web App

Once the Backend Pod has trained and saved the model:

```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`.

**Features:**
- Upload a video file
- See prediction: "Viral" or "Not Viral"
- View confidence score
- Understand feature importance

---

## 🔀 Git Workflow

### Backend Pod Workflow
```bash
# Person 1 or Person 2 (on data-model branch)
git checkout data-model
git pull origin data-model  # Always pull before starting work

# Make changes to your assigned file
git add src/your_file.py
git commit -m "Implemented feature extraction for audio"
git push origin data-model
```

### Frontend Dev Workflow
```bash
# On webapp branch
git checkout webapp
git pull origin webapp

# Make changes to streamlit_app.py
git add app/streamlit_app.py
git commit -m "Added video upload UI"
git push origin webapp
```

### Merging to Main (End of Project)
```bash
# Merge backend work
git checkout main
git merge data-model

# Merge frontend work
git merge webapp

# Push final version
git push origin main
```

---

## 📦 Dependencies

All dependencies are listed in `requirements.txt`:

- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning models
- **librosa**: Audio feature extraction
- **opencv-python**: Video/frame processing
- **streamlit**: Web application framework
- **joblib**: Model serialization

Install with:
```bash
pip install -r requirements.txt
```

---

## 🔍 Key Files & Ownership

| File | Owner | Purpose |
|------|-------|---------|
| `src/preprocess.py` | Backend Person 1 | Feature extraction |
| `src/train_model.py` | Backend Person 2 | Model training |
| `src/predict.py` | Backend (shared) | Inference functions |
| `src/utils.py` | All | Helper functions |
| `app/streamlit_app.py` | Frontend Dev | Web interface |
| `notebooks/EDA.ipynb` | Backend (optional) | Data exploration |

---

## 🌟 Future Work

- **Browser Extension:** Integrate ViralVision as a Chrome extension to predict virality directly on TikTok/YouTube
- **Dataset Scaling:** Collect more videos to improve model accuracy
- **Deep Learning:** Experiment with CNNs for video frame analysis
- **Real-time Predictions:** Deploy to cloud (AWS/GCP) for production use
- **Feature Engineering:** Add more sophisticated features (face detection, text overlays, hashtag analysis)

---

## 🐛 Troubleshooting

### Issue: Model file not found
**Solution:** Make sure Backend Pod has trained and saved `models/model.pkl`

### Issue: Import errors in Streamlit
**Solution:** Check that `sys.path` is correctly set in `streamlit_app.py`

### Issue: Video files too large for Git
**Solution:** Never commit videos. Share via Google Drive or Dropbox. Check `.gitignore`.

### Issue: Merge conflicts
**Solution:** Each team member should work on their assigned files only. Communicate before editing shared files.

---

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## 🙌 Acknowledgments

- MAIS202 Teaching Team
- Open-source libraries: scikit-learn, librosa, OpenCV, Streamlit
- Dataset sources: [Add your data sources here]

---

**Happy Coding! 🚀**

For questions, contact your team members via Slack/Discord.

