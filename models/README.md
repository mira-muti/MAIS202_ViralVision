# Models Directory

**Purpose:** Store trained machine learning models.

**Owner:** Backend Pod (Person 2)

**Notes:**
- This directory contains serialized model files
- Models are saved using `joblib` or `pickle`
- Files here are ignored by Git (too large and binary)
- Share trained models via Google Drive if needed

**Expected Files:**
- `model.pkl` - The primary trained model (RandomForest or Naive Bayes)
- `random_forest.pkl` - Random Forest model
- `naive_bayes.pkl` - Naive Bayes model
- `scaler.pkl` - Fitted scaler for feature normalization (if used)
- `model_metadata.json` - Model training metadata (accuracy, date, features used)

**Usage:**
Models are loaded by `src/predict.py` for inference in the Streamlit app.

**Training:**
```bash
python src/train_model.py
```

This will create `models/model.pkl` automatically.

