"""
Training pipeline for ViralVision engagement prediction model.

Trains RandomForest and GradientBoosting models with automatic
overfitting detection and feature standardization.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings

# Paths
project_root = Path(__file__).parent.parent
DATASET_PATH = project_root / "data" / "processed" / "grwm_dataset.csv"

MODELS_DIR = project_root / "backend" / "models"
MODEL_PATH = MODELS_DIR / "grwm_model.pkl"
ENCODER_PATH = MODELS_DIR / "grwm_encoder.pkl"
SCALER_PATH = MODELS_DIR / "grwm_scaler.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "grwm_feature_names.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    """
    Load and clean the processed dataset.
    
    Returns:
        DataFrame with cleaned data
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    try:
        df = pd.read_csv(DATASET_PATH, comment='#')
    except Exception as e:
        raise ValueError(f"Error reading dataset: {DATASET_PATH}\nError: {str(e)}")
    
    if df.empty:
        raise ValueError(f"Dataset is empty: {DATASET_PATH}")
    
    # Check required columns
    required_columns = [
        'caption_length', 'hashtag_count', 'fft_max_freq', 'fft_max_amp',
        'avg_brightness', 'color_std_dev', 'motion_intensity', 'scene_change_rate',
        'hue_entropy', 'face_present', 'text_overlay_present',
        'rms_energy', 'zcr', 'spectral_centroid', 'spectral_rolloff',
        'niche', 'label'
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Drop rows with missing values
    df = df.dropna(subset=required_columns)
    
    # Ensure label is numeric (0 or 1)
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df[df['label'].isin([0, 1])]
    
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Label distribution:")
    print(f"  Low (0): {(df['label'] == 0).sum()}")
    print(f"  High (1): {(df['label'] == 1).sum()}")
    if 'niche' in df.columns:
        print(f"Niche distribution:")
        print(df['niche'].value_counts())
    
    return df


def build_feature_matrix(df):
    """
    Build feature matrix with proper encoding and scaling.
    
    Args:
        df: DataFrame with dataset
        
    Returns:
        X: Feature matrix
        y: Labels
        encoder: Fitted OneHotEncoder
        scaler: Fitted StandardScaler
        feature_names: List of feature names in order
    """
    # Numeric features (exclude engagement_ratio for inference)
    numeric_cols = [
        'engagement_ratio',  # Included for training, set to 0 at inference
        'caption_length',
        'hashtag_count',
        'avg_brightness',
        'color_std_dev',
        'motion_intensity',
        'scene_change_rate',
        'hue_entropy',
        'face_present',
        'text_overlay_present',
        'rms_energy',
        'zcr',
        'spectral_centroid',
        'spectral_rolloff',
        'fft_max_freq',
        'fft_max_amp',
    ]
    
    # Ensure engagement_ratio exists
    if 'engagement_ratio' not in df.columns:
        df['engagement_ratio'] = 0.0
    
    numeric_features = df[numeric_cols].copy()
    
    # Encode niche using OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    niche_encoded = encoder.fit_transform(df[['niche']])
    niche_feature_names = encoder.get_feature_names_out(['niche'])
    niche_df = pd.DataFrame(niche_encoded, columns=niche_feature_names, index=df.index)
    
    # Combine features
    X = pd.concat([numeric_features, niche_df], axis=1)
    y = df['label'].values
    
    # Standardize numeric features
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    feature_names = X_scaled.columns.tolist()
    
    print(f"\nFeature matrix shape: {X_scaled.shape}")
    print(f"Features: {len(feature_names)}")
    
    return X_scaled, y, encoder, scaler, feature_names


def test_shuffled_labels(X_train, y_train, X_test, y_test):
    """
    Test if model learns noise (shuffled label test).
    
    Returns:
        True if model behaves correctly (accuracy ~50%), False if overfitting
    """
    print("\n" + "-" * 60)
    print("Shuffled Label Test (Overfitting Detection)")
    print("-" * 60)
    
    # Shuffle labels
    y_train_shuffled = y_train.copy()
    np.random.seed(42)
    np.random.shuffle(y_train_shuffled)
    
    # Train temporary model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        temp_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        temp_model.fit(X_train, y_train_shuffled)
    
    # Evaluate on real test labels
    y_pred = temp_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Shuffled-label model accuracy: {accuracy:.4f}")
    print(f"Expected: ~0.50 (random chance)")
    
    if accuracy > 0.55:
        print("⚠️ WARNING: Overfitting detected — model learns noise!")
        return False
    elif accuracy < 0.45:
        print("⚠️ Unexpectedly low accuracy (possible bug)")
        return False
    else:
        print("✔️ Model behaves correctly on shuffled labels")
        return True


def test_class_collapse(y_pred):
    """
    Test if model predicts only one class.
    
    Returns:
        True if both classes predicted, False if collapsed
    """
    unique = set(y_pred)
    if len(unique) == 1:
        print("❌ MODEL COLLAPSED — predicting only one class!")
        return False
    else:
        print("✔️ Model predicts both classes")
        return True


def train_models(X_train, y_train, X_test, y_test):
    """
    Train multiple models and select the best.
    
    Returns:
        Best model based on F1-score
    """
    print("\n" + "=" * 60)
    print("Training Models")
    print("=" * 60)
    
    models = {}
    results = {}
    
    # Train Random Forest
    print("\n[1] Training RandomForestClassifier...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
        models['RandomForest'] = rf_model
        results['RandomForest'] = rf_f1
        print(f"    F1-score: {rf_f1:.4f}")
    
    # Train Gradient Boosting
    print("\n[2] Training GradientBoostingClassifier...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            learning_rate=0.1
        )
        gb_model.fit(X_train, y_train)
        gb_pred = gb_model.predict(X_test)
        gb_f1 = f1_score(y_test, gb_pred, zero_division=0)
        models['GradientBoosting'] = gb_model
        results['GradientBoosting'] = gb_f1
        print(f"    F1-score: {gb_f1:.4f}")
    
    # Select best model
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    best_f1 = results[best_model_name]
    
    print(f"\n{'=' * 60}")
    print(f"Best Model: {best_model_name} (F1-score: {best_f1:.4f})")
    print(f"{'=' * 60}")
    
    return best_model, best_model_name


def evaluate(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance.
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluation: {model_name}")
    print(f"{'=' * 60}")
    
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
    
    # Test class collapse
    test_class_collapse(y_pred)
    
    return y_pred


def save_artifacts(model, encoder, scaler, feature_names):
    """
    Save model, encoder, scaler, and feature names.
    """
    print(f"\n{'=' * 60}")
    print("Saving Artifacts")
    print(f"{'=' * 60}")
    
    joblib.dump(model, MODEL_PATH)
    print(f"✓ Model saved: {MODEL_PATH}")
    
    joblib.dump(encoder, ENCODER_PATH)
    print(f"✓ Encoder saved: {ENCODER_PATH}")
    
    joblib.dump(scaler, SCALER_PATH)
    print(f"✓ Scaler saved: {SCALER_PATH}")
    
    with open(FEATURE_NAMES_PATH, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"✓ Feature names saved: {FEATURE_NAMES_PATH}")
    
    print(f"\nAll artifacts saved successfully!")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("ViralVision Model Training Pipeline")
    print("=" * 60)
    
    # Load dataset
    df = load_dataset()
    
    # ============================================================
    # TEMPORARY QUICK FIX — CREATE SYNTHETIC HIGH SAMPLES
    # ============================================================
    
    print("⚠️  TEMP FIX ENABLED: Generating synthetic HIGH samples to prevent class collapse")
    
    high_class = df[df['label'] == 1]
    
    # If we have fewer than 40 high samples, generate synthetic ones
    if len(high_class) < 40:
        import numpy as np
        synthetic = []
        needed = 40 - len(high_class)
        
        print(f"   → Existing HIGH samples: {len(high_class)}")
        print(f"   → Generating {needed} synthetic HIGH samples...")
        
        for i in range(needed):
            # sample one real high row
            row = high_class.sample(1).iloc[0].copy()
            
            # columns to add noise to (only if they exist in the dataset)
            possible_numeric_cols = [
                'engagement_ratio','caption_length','hashtag_count',
                'fft_max_freq','fft_max_amp','rms_energy','zcr',
                'spectral_centroid','spectral_rolloff','tempo',
                'avg_brightness','avg_color_variance','motion_intensity',
                'color_std_dev'  # Alternative name
            ]
            
            # Filter to only columns that exist in the dataset
            numeric_cols = [col for col in possible_numeric_cols if col in row.index]
            
            # add small gaussian noise
            for col in numeric_cols:
                if pd.notna(row[col]) and pd.api.types.is_numeric_dtype(type(row[col])):
                    val = row[col]
                    noise = np.random.normal(0, 0.05) * (val if val != 0 else 1)
                    row[col] = max(0, val + noise)
            
            # keep label as HIGH
            row['label'] = 1
            synthetic.append(row)
        
        df = pd.concat([df, pd.DataFrame(synthetic)], ignore_index=True)
        print(f"✔ Added {len(synthetic)} synthetic HIGH samples")
    else:
        print("✔ Enough HIGH samples — no synthetic data needed")
    
    # ============================================================
    # END TEMP FIX
    # ============================================================
    
    # Build feature matrix
    X, y, encoder, scaler, feature_names = build_feature_matrix(df)
    
    # Train-test split
    print(f"\n{'=' * 60}")
    print("Train-Test Split")
    print(f"{'=' * 60}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set:     {X_test.shape[0]} samples")
    
    # Overfitting detection: shuffled label test
    test_shuffled_labels(X_train, y_train, X_test, y_test)
    
    # Train models
    best_model, best_model_name = train_models(X_train, y_train, X_test, y_test)
    
    # Evaluate best model
    evaluate(best_model, X_test, y_test, best_model_name)
    
    # Save artifacts
    save_artifacts(best_model, encoder, scaler, feature_names)
    
    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
