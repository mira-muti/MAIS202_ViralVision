"""
File: train_model.py

Purpose: 
    Train machine learning models (RandomForest and Naive Bayes) on the processed feature dataset.
    Evaluates models and saves the best performing model based on F1-score.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from src.extract_audio_features import extract_fft_features
import joblib
from pathlib import Path


def load_dataset(csv_path):
    """
    Load and prepare the processed dataset.
    
    Args:
        csv_path (str): Path to final_dataset.csv
        
    Returns:
        tuple: (X, y) where X is feature matrix and y is encoded labels (0/1)
    """
    df = pd.read_csv(csv_path)
    
    # Separate features and target
    feature_cols = ['engagement_ratio', 'caption_length', 'hashtag_count', 
                    'fft_max_freq', 'fft_max_amp', 'niche']
    X = df[feature_cols].copy()
    y = df['label'].copy()
    
    # Encode labels: High=1, Low=0
    y = y.map({'High': 1, 'Low': 0})
    
    # One-hot encode niche
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    niche_encoded = encoder.fit_transform(X[['niche']])
    niche_feature_names = encoder.get_feature_names_out(['niche'])
    niche_df = pd.DataFrame(niche_encoded, columns=niche_feature_names, index=X.index)
    
    # Combine numeric features with encoded niche
    numeric_features = X[['engagement_ratio', 'caption_length', 'hashtag_count', 
                          'fft_max_freq', 'fft_max_amp']]
    X_processed = pd.concat([numeric_features, niche_df], axis=1)
    
    return X_processed, y, encoder


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        RandomForestClassifier: Trained model
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    return model


def train_naive_bayes(X_train, y_train):
    """
    Train a Gaussian Naive Bayes classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        GaussianNB: Trained model
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return metrics


def save_model(model, encoder, model_path):
    """
    Save trained model and encoder to disk.
    
    Args:
        model: Trained classifier
        encoder: OneHotEncoder used for niche encoding
        model_path: Path to save the model
    """
    model_dir = Path(model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, model_path)
    
    # Save encoder alongside model
    encoder_path = model_path.replace('.pkl', '_encoder.pkl')
    joblib.dump(encoder, encoder_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Encoder saved to: {encoder_path}")


def main():
    """
    Complete training pipeline.
    """
    print("=" * 60)
    print("Model Training Pipeline")
    print("=" * 60)
    
    # Load dataset
    print("\n[1] Loading dataset...")
    csv_path = "data/processed/final_dataset.csv"
    X, y, encoder = load_dataset(csv_path)
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"Label distribution: {y.value_counts().to_dict()}")
    
    # Split data
    print("\n[2] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train Random Forest
    print("\n[3] Training Random Forest classifier...")
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    
    # Train Naive Bayes
    print("[4] Training Naive Bayes classifier...")
    nb_model = train_naive_bayes(X_train, y_train)
    nb_metrics = evaluate_model(nb_model, X_test, y_test)
    
    # Compare models
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print("\nRandom Forest:")
    print(f"  Accuracy:  {rf_metrics['accuracy']:.4f}")
    print(f"  Precision: {rf_metrics['precision']:.4f}")
    print(f"  Recall:    {rf_metrics['recall']:.4f}")
    print(f"  F1-Score:  {rf_metrics['f1_score']:.4f}")
    print(f"  Confusion Matrix:\n{rf_metrics['confusion_matrix']}")
    
    print("\nNaive Bayes:")
    print(f"  Accuracy:  {nb_metrics['accuracy']:.4f}")
    print(f"  Precision: {nb_metrics['precision']:.4f}")
    print(f"  Recall:    {nb_metrics['recall']:.4f}")
    print(f"  F1-Score:  {nb_metrics['f1_score']:.4f}")
    print(f"  Confusion Matrix:\n{nb_metrics['confusion_matrix']}")
    
    # Select best model based on F1-score
    print("\n" + "=" * 60)
    print("Selecting Best Model")
    print("=" * 60)
    
    if rf_metrics['f1_score'] >= nb_metrics['f1_score']:
        best_model = rf_model
        best_name = "Random Forest"
        best_metrics = rf_metrics
    else:
        best_model = nb_model
        best_name = "Naive Bayes"
        best_metrics = nb_metrics
    
    print(f"Best model: {best_name} (F1-Score: {best_metrics['f1_score']:.4f})")
    
    # Save best model
    print("\n[5] Saving best model...")
    model_path = "models/model.pkl"
    save_model(best_model, encoder, model_path)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
