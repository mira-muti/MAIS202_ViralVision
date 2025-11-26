"""
Training pipeline for ViralVision engagement prediction model.

This script trains a Random Forest and Gaussian Naive Bayes classifier
on the processed dataset and saves the best model for inference.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

# Paths
project_root = Path(__file__).parent.parent
DATASET_PATH = project_root / "data" / "processed" / "final_dataset.csv"

# GRWM-specific model artifacts (stored under backend/models)
MODELS_DIR = project_root / "backend" / "models"
MODEL_PATH = MODELS_DIR / "grwm_model.pkl"
ENCODER_PATH = MODELS_DIR / "grwm_encoder.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "grwm_feature_names.json"

# Ensure models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    """
    Load and clean the processed dataset.
    
    Returns:
        DataFrame with cleaned data
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    # Check if file has actual data (not just comments)
    with open(DATASET_PATH, 'r') as f:
        content = f.read()
        # Find lines that aren't comments or empty
        data_lines = [line.strip() for line in content.split('\n') 
                      if line.strip() and not line.strip().startswith('#')]
        if not data_lines:
            raise ValueError(
                f"Dataset file contains only comments, no data: {DATASET_PATH}\n"
                f"Please run preprocessing script to generate final_dataset.csv with actual data.\n"
                f"Expected columns: engagement_ratio, caption_length, hashtag_count, "
                f"fft_max_freq, fft_max_amp, niche, label"
            )
    
    # Try to read CSV, skipping comment lines
    try:
        df = pd.read_csv(DATASET_PATH, comment='#')
    except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as e:
        raise ValueError(
            f"Error reading dataset: {DATASET_PATH}\n"
            f"Error: {str(e)}\n"
            f"Please ensure the file is a valid CSV with the required columns."
        )
    
    if df.empty:
        raise ValueError(
            f"Dataset is empty after loading: {DATASET_PATH}\n"
            f"Please run preprocessing script to generate final_dataset.csv with data."
        )
    
    # Check if required columns exist
    required_columns = ['caption_length', 'hashtag_count', 'fft_max_freq', 'fft_max_amp', 'niche', 'label']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}. Found columns: {df.columns.tolist()}")
    
    # Drop rows with missing values in critical columns
    df = df.dropna(subset=required_columns)
    
    # Remove rows with invalid numeric values
    numeric_cols = ['caption_length', 'hashtag_count', 'fft_max_freq', 'fft_max_amp', 'engagement_ratio']
    for col in numeric_cols:
        if col in df.columns:
            df = df[pd.to_numeric(df[col], errors='coerce').notna()]
    
    # Ensure label is numeric (0 or 1)
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df[df['label'].isin([0, 1])]

    # Filter to GRWM-related niches only (exclude pure music)
    allowed_niches = {
        'GRWM', 'grwm',
        'OOTD', 'ootd',
        'fyp', 'FYP',
        'summervibes', 'summervibe', 'summer',
        'food', 'Food',
        'DIYProjects', 'diy', 'DIY',
        'dance', 'Dance',
    }
    before_niche = len(df)
    df = df[df['niche'].astype(str).isin(allowed_niches)]
    after_niche = len(df)
    
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Filtered out {before_niche - after_niche} rows not in GRWM-related niches")
    print(f"Label distribution:")
    print(df['label'].value_counts().sort_index())
    print(f"  Low (0): {(df['label'] == 0).sum()}")
    print(f"  High (1): {(df['label'] == 1).sum()}")
    
    return df


def build_feature_matrix(df):
    """
    Build feature matrix with proper encoding.
    
    Matches predict.py structure: includes engagement_ratio (set to 0 at inference)
    but uses actual values during training.
    
    Args:
        df: DataFrame with dataset
        
    Returns:
        X: Feature matrix
        y: Labels
        encoder: Fitted OneHotEncoder
        feature_names: List of feature names in order
    """
    # Verify required columns exist
    required_cols = ['caption_length', 'hashtag_count', 'fft_max_freq', 'fft_max_amp', 'niche']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Extract numeric features (include engagement_ratio for training compatibility)
    # Note: engagement_ratio is included in training but set to 0 at inference
    numeric_cols = ['engagement_ratio', 'caption_length', 'hashtag_count', 'fft_max_freq', 'fft_max_amp']
    if 'engagement_ratio' not in df.columns:
        # If engagement_ratio doesn't exist, create it as 0 (for compatibility)
        df['engagement_ratio'] = 0.0
    
    numeric_features = df[numeric_cols].copy()
    
    # Encode niche using OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    niche_encoded = encoder.fit_transform(df[['niche']])
    niche_feature_names = encoder.get_feature_names_out(['niche'])
    niche_df = pd.DataFrame(niche_encoded, columns=niche_feature_names, index=df.index)
    
    # Combine features (matches predict.py order)
    X = pd.concat([numeric_features, niche_df], axis=1)
    
    # Extract labels
    y = df['label'].values
    
    # Get feature names in order
    feature_names = X.columns.tolist()
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Feature names ({len(feature_names)}):")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i}. {name}")
    
    return X, y, encoder, feature_names


def train_models(X_train, y_train, X_test, y_test):
    """
    Train Random Forest and Gaussian Naive Bayes models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Best model based on F1-score
    """
    print("\n" + "="*60)
    print("Training Models")
    print("="*60)
    
    models = {}
    results = {}
    
    # Train Random Forest
    print("\n[1] Training RandomForestClassifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_f1 = f1_score(y_test, rf_pred)
    models['RandomForest'] = rf_model
    results['RandomForest'] = rf_f1
    print(f"    F1-score: {rf_f1:.4f}")
    
    # Train Gaussian Naive Bayes
    print("\n[2] Training GaussianNB...")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_f1 = f1_score(y_test, nb_pred)
    models['GaussianNB'] = nb_model
    results['GaussianNB'] = nb_f1
    print(f"    F1-score: {nb_f1:.4f}")
    
    # Select best model
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    best_f1 = results[best_model_name]
    
    print(f"\n{'='*60}")
    print(f"Best Model: {best_model_name} (F1-score: {best_f1:.4f})")
    print(f"{'='*60}")
    
    return best_model, best_model_name


def evaluate(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for display
    """
    print(f"\n{'='*60}")
    print(f"Evaluation: {model_name}")
    print(f"{'='*60}")
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                Predicted")
    print(f"              Low    High")
    print(f"  Actual Low   {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"        High   {cm[1,0]:4d}   {cm[1,1]:4d}")


def save_artifacts(model, encoder, feature_names):
    """
    Save model, encoder, and feature names.
    
    Args:
        model: Trained model
        encoder: Fitted OneHotEncoder
        feature_names: List of feature names
    """
    print(f"\n{'='*60}")
    print("Saving Artifacts")
    print(f"{'='*60}")
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"✓ Model saved: {MODEL_PATH}")
    
    # Save encoder
    joblib.dump(encoder, ENCODER_PATH)
    print(f"✓ Encoder saved: {ENCODER_PATH}")
    
    # Save feature names
    with open(FEATURE_NAMES_PATH, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"✓ Feature names saved: {FEATURE_NAMES_PATH}")
    
    print(f"\nAll artifacts saved successfully!")


def main():
    """Main training pipeline."""
    print("="*60)
    print("ViralVision Model Training Pipeline")
    print("="*60)
    
    # Load dataset
    df = load_dataset()
    
    # Build feature matrix
    X, y, encoder, feature_names = build_feature_matrix(df)
    
    # Train-test split
    print(f"\n{'='*60}")
    print("Train-Test Split")
    print(f"{'='*60}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set:     {X_test.shape[0]} samples")
    
    # Train models
    best_model, best_model_name = train_models(X_train, y_train, X_test, y_test)
    
    # Evaluate best model
    evaluate(best_model, X_test, y_test, best_model_name)
    
    # Save artifacts
    save_artifacts(best_model, encoder, feature_names)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

