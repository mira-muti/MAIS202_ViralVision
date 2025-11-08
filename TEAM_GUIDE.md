# 👥 Team Member Quick Start Guide

## 🎯 Backend Pod - Person 1 (Feature Extraction)

### Your Responsibilities
- Extract audio features from videos
- Extract visual features from videos
- Build the final dataset CSV

### Your Files
**Main file:** `src/preprocess.py` ✅ Already set up with function templates

**Testing:** `notebooks/feature_test.ipynb` ✅ Test your extraction functions here

### Your Branch
```bash
git checkout -b data-model
```

### Next Steps
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add sample videos** to `data/raw/` (organize in `viral/` and `not_viral/` subdirectories)

3. **Implement functions in `src/preprocess.py`:**
   - `extract_audio_features()` - Use librosa to extract audio features
   - `extract_visual_features()` - Use OpenCV to extract visual features
   - `build_feature_dataset()` - Process all videos and create CSV

4. **Test your functions** in `notebooks/feature_test.ipynb`

5. **Run the full pipeline:**
   ```bash
   python src/preprocess.py
   ```

6. **Verify output:** Check that `data/final_dataset.csv` is created correctly

### Coordinate with Person 2
- **Agree on feature names** (column names in CSV)
- **Document all features** in comments
- **Share sample dataset** once features are extracted

---

## 🎯 Backend Pod - Person 2 (Model Training)

### Your Responsibilities
- Train RandomForest and Naive Bayes models
- Evaluate model performance
- Save the best model
- Implement prediction functions

### Your Files
**Main files:** 
- `src/train_model.py` ✅ Model training pipeline
- `src/predict.py` ✅ Inference functions (shared with Person 1)

**Testing:** `notebooks/EDA.ipynb` ✅ Explore the dataset

### Your Branch
```bash
git checkout -b data-model
```

### Next Steps
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Wait for Person 1** to create `data/final_dataset.csv`
   - Or create a dummy CSV to start coding

3. **Implement functions in `src/train_model.py`:**
   - `load_dataset()` - Load the CSV
   - `train_random_forest()` - Train RF classifier
   - `train_naive_bayes()` - Train NB classifier
   - `evaluate_model()` - Calculate metrics
   - `save_model()` - Save to `models/model.pkl`

4. **Explore the data** in `notebooks/EDA.ipynb`

5. **Train the model:**
   ```bash
   python src/train_model.py
   ```

6. **Implement inference in `src/predict.py`:**
   - `load_trained_model()` - Load model.pkl
   - `predict_single_video()` - Predict for one video
   - `predict_with_confidence()` - Return probabilities

7. **Test predictions:**
   ```bash
   python src/predict.py
   ```

### Coordinate with Person 1
- **Confirm feature names** match what Person 1 provides
- **Document model requirements** (which features are needed)

### Coordinate with Frontend Dev
- **Share function signatures** for `predict.py`
- **Document input/output formats** clearly
- **Test that predictions work** before frontend integration

---

## 🎯 Frontend Dev (Beginner)

### Your Responsibilities
- Build a user-friendly Streamlit web app
- Allow users to upload videos
- Display predictions and confidence scores
- Create visualizations

### Your File
**Main file:** `app/streamlit_app.py` ✅ Already set up with function templates

### Your Branch
```bash
git checkout -b webapp
```

### Next Steps
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Learn Streamlit basics** (if needed):
   - [Streamlit Documentation](https://docs.streamlit.io/)
   - Simple tutorial: File upload, buttons, text display

3. **Implement functions in `app/streamlit_app.py`:**
   - `display_header()` - Show app title and description
   - `upload_video_section()` - Handle video uploads
   - `display_prediction_results()` - Show prediction with emoji and confidence
   - `main()` - Wire everything together

4. **Test the app locally:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

5. **Add features progressively:**
   - Start simple: title + file upload
   - Add prediction display
   - Add confidence bars
   - Add explanations and visualizations

### Coordinate with Backend Pod
- **Ask for help** with `predict.py` functions if unclear
- **Request a sample model** (`models/model.pkl`) for testing
- **Document any issues** with backend functions

### UI Tips
- Use `st.title()`, `st.header()`, `st.subheader()` for text
- Use `st.file_uploader()` for video upload
- Use `st.progress()` for confidence bars
- Use `st.success()` / `st.warning()` for predictions
- Use `st.sidebar` for additional info

---

## 🤝 Team Coordination

### Communication
- **Never edit the same file simultaneously**
- **Pull before you start working each day:**
  ```bash
  git pull origin your-branch-name
  ```
- **Commit and push frequently:**
  ```bash
  git add your-file.py
  git commit -m "Clear description of changes"
  git push origin your-branch-name
  ```

### Shared Files
- **`src/utils.py`** - All team members can add helper functions here
- **`src/predict.py`** - Backend pair shares this file (Person 2 leads)

### File Ownership
| File | Owner | Can Others Edit? |
|------|-------|------------------|
| `src/preprocess.py` | Person 1 | No |
| `src/train_model.py` | Person 2 | No |
| `src/predict.py` | Person 2 (lead) | Person 1 can help |
| `src/utils.py` | Shared | Yes |
| `app/streamlit_app.py` | Frontend Dev | No |
| `notebooks/EDA.ipynb` | Backend (shared) | Yes |
| `notebooks/feature_test.ipynb` | Person 1 | Person 2 can view |

### Milestones
**Week 1:**
- [ ] Person 1: Feature extraction working
- [ ] Person 1: `data/final_dataset.csv` created
- [ ] Person 2: Model training pipeline working
- [ ] Person 2: `models/model.pkl` saved
- [ ] Person 2: `predict.py` functions implemented
- [ ] Frontend Dev: Basic Streamlit app structure

**Week 2:**
- [ ] Backend: Model evaluation and tuning
- [ ] Backend: Documentation and testing
- [ ] Frontend: Complete UI with all features
- [ ] Team: Integration testing
- [ ] Team: Final demo preparation

### Questions?
- Check `README.md` for detailed setup instructions
- Review file headers for function descriptions
- Ask your teammates on Slack/Discord

---

**Good luck! 🚀**

