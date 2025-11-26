"""
Model Integrity Test Suite for ViralVision.

Tests both GRWM and OOTD models for:
- Overfitting
- Class collapse
- Hallucinated probabilities
- Sensitivity to noise
- Missing files
"""

import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

project_root = backend_dir.parent

# Paths
DATASET_PATH = project_root / "data" / "processed" / "final_dataset.csv"
MODELS_DIR = project_root / "backend" / "models"

GRWM_MODEL_PATH = MODELS_DIR / "grwm_model.pkl"
GRWM_ENCODER_PATH = MODELS_DIR / "grwm_encoder.pkl"
GRWM_FEATURE_NAMES_PATH = MODELS_DIR / "grwm_feature_names.json"

OOTD_MODEL_PATH = MODELS_DIR / "ootd_model.pkl"
OOTD_ENCODER_PATH = MODELS_DIR / "ootd_encoder.pkl"
OOTD_FEATURE_NAMES_PATH = MODELS_DIR / "ootd_feature_names.json"


def print_section(title: str):
    """Print a beautiful section divider."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60 + "\n")


def load_models_and_dataset():
    """Load both models, encoders, and the dataset."""
    print_section("LOADING MODELS AND DATASET")
    
    results = {}
    
    # Check dataset
    if not DATASET_PATH.exists():
        print("❌ Dataset not found:", DATASET_PATH)
        return None
    
    try:
        df = pd.read_csv(DATASET_PATH, comment='#')
        print(f"✔️ Dataset loaded: {len(df)} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None
    
    # Check GRWM model
    if not GRWM_MODEL_PATH.exists():
        print(f"❌ GRWM model not found: {GRWM_MODEL_PATH}")
        results['grwm'] = None
    else:
        try:
            grwm_model = joblib.load(GRWM_MODEL_PATH)
            grwm_encoder = joblib.load(GRWM_ENCODER_PATH)
            print(f"✔️ GRWM model loaded: {type(grwm_model).__name__}")
            results['grwm'] = {
                'model': grwm_model,
                'encoder': grwm_encoder,
                'name': 'GRWM'
            }
        except Exception as e:
            print(f"❌ Error loading GRWM model: {e}")
            results['grwm'] = None
    
    # Check OOTD model
    if not OOTD_MODEL_PATH.exists():
        print(f"❌ OOTD model not found: {OOTD_MODEL_PATH}")
        results['ootd'] = None
    else:
        try:
            ootd_model = joblib.load(OOTD_MODEL_PATH)
            ootd_encoder = joblib.load(OOTD_ENCODER_PATH)
            print(f"✔️ OOTD model loaded: {type(ootd_model).__name__}")
            results['ootd'] = {
                'model': ootd_model,
                'encoder': ootd_encoder,
                'name': 'OOTD'
            }
        except Exception as e:
            print(f"❌ Error loading OOTD model: {e}")
            results['ootd'] = None
    
    results['dataset'] = df
    return results


def build_feature_matrix(df, encoder):
    """Build feature matrix matching training pipeline."""
    numeric_cols = [
        'engagement_ratio', 'caption_length', 'hashtag_count',
        'avg_brightness', 'color_std_dev', 'motion_intensity', 'scene_change_rate',
        'hue_entropy', 'face_present', 'text_overlay_present',
        'rms_energy', 'zcr', 'spectral_centroid', 'spectral_rolloff',
        'fft_max_freq', 'fft_max_amp',
    ]
    
    # Ensure all columns exist
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    numeric_features = df[numeric_cols].copy()
    
    # Encode niche
    niche_encoded = encoder.transform(df[['niche']])
    niche_feature_names = encoder.get_feature_names_out(['niche'])
    niche_df = pd.DataFrame(niche_encoded, columns=niche_feature_names, index=df.index)
    
    X = pd.concat([numeric_features, niche_df], axis=1)
    y = df['label'].values
    
    return X, y


def test_standard_metrics(model_info, df):
    """Test 1: Standard train/test accuracy metrics."""
    print_section(f"TEST 1: STANDARD METRICS - {model_info['name']}")
    
    model = model_info['model']
    encoder = model_info['encoder']
    niche = model_info['name']
    
    # Filter to niche
    df_niche = df[df['niche'].astype(str).str.upper() == niche.upper()].copy()
    
    if len(df_niche) < 10:
        print(f"⚠️ Not enough data for {niche}: {len(df_niche)} rows")
        return None
    
    # Build features
    X, y = build_feature_matrix(df_niche, encoder)
    
    # Align with model features if needed
    if hasattr(model, 'feature_names_in_'):
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Test set size: {len(X_test)}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Low    High")
    print(f"  Actual Low   {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"        High   {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'X_test': X_test
    }


def test_class_collapse(model_info, test_results):
    """Test 2: Check for class collapse."""
    print_section(f"TEST 2: CLASS COLLAPSE CHECK - {model_info['name']}")
    
    if test_results is None:
        print("⚠️ Skipping - no test results available")
        return False
    
    y_pred = test_results['y_pred']
    unique_predictions = set(y_pred)
    
    print(f"Unique predictions: {unique_predictions}")
    
    if len(unique_predictions) == 1:
        print("❌ MODEL COLLAPSED — predicting only one class!")
        return False
    else:
        print("✔️ Model predicts both classes")
        return True


def test_probability_sanity(model_info, test_results):
    """Test 3: Probability sanity check."""
    print_section(f"TEST 3: PROBABILITY SANITY - {model_info['name']}")
    
    if test_results is None:
        print("⚠️ Skipping - no test results available")
        return False
    
    y_proba = test_results['y_proba']
    y_test = test_results['y_test']
    
    # Sample 10 random instances
    n_samples = min(10, len(y_proba))
    indices = np.random.choice(len(y_proba), n_samples, replace=False)
    
    print(f"Sampling {n_samples} random predictions:\n")
    issues = []
    
    for idx in indices:
        prob_high = y_proba[idx][1] if y_proba.shape[1] > 1 else y_proba[idx][0]
        actual = "High" if y_test[idx] == 1 else "Low"
        predicted = "High" if y_proba[idx].argmax() == 1 else "Low"
        
        print(f"  Instance {idx}: prob_high={prob_high:.3f}, actual={actual}, pred={predicted}")
        
        # Check for extreme probabilities
        if prob_high > 0.99 and y_test[idx] == 0:
            issues.append(f"Extreme confidence ({prob_high:.3f}) for Low class")
        if prob_high < 0.01 and y_test[idx] == 1:
            issues.append(f"Extreme confidence ({prob_high:.3f}) for High class")
    
    # Check all probabilities are valid
    all_probs = y_proba.flatten()
    if np.any(all_probs < 0) or np.any(all_probs > 1):
        print("❌ Invalid probabilities detected (outside [0, 1])")
        return False
    
    if issues:
        print(f"\n⚠️ Found {len(issues)} probability issues")
        for issue in issues[:5]:  # Show first 5
            print(f"  - {issue}")
        return False
    else:
        print("\n✔️ Probabilities look sane")
        return True


def test_zero_features(model_info, encoder):
    """Test 4: Zero-feature test."""
    print_section(f"TEST 4: ZERO-FEATURE TEST - {model_info['name']}")
    
    model = model_info['model']
    niche = model_info['name']
    
    # Build zero feature vector
    if hasattr(model, 'feature_names_in_'):
        n_features = len(model.feature_names_in_)
        feature_names = model.feature_names_in_
    else:
        # Estimate from encoder
        n_features = 16 + len(encoder.get_feature_names_out(['niche']))
        feature_names = None
    
    zero_input = np.zeros((1, n_features))
    
    # Create DataFrame with proper structure
    if feature_names is not None:
        zero_df = pd.DataFrame(zero_input, columns=feature_names)
    else:
        zero_df = pd.DataFrame(zero_input)
    
    # Predict
    try:
        proba = model.predict_proba(zero_df)[0]
        prob_high = proba[1] if len(proba) > 1 else proba[0]
        label = model.predict(zero_df)[0]
        
        print(f"Zero-input prediction:")
        print(f"  Label: {'High' if label == 1 else 'Low'}")
        print(f"  Probability (High): {prob_high:.4f}")
        
        if prob_high > 0.80:
            print("⚠️ Model hallucinating on zero-input! (prob > 0.80)")
            return False
        else:
            print("✔️ Model handles zero-input reasonably")
            return True
    except Exception as e:
        print(f"⚠️ Error predicting zero-input: {e}")
        return False


def test_shuffled_labels(model_info, df):
    """Test 5: Shuffled label test (gold standard)."""
    print_section(f"TEST 5: SHUFFLED LABEL TEST - {model_info['name']}")
    
    model = model_info['model']
    encoder = model_info['encoder']
    niche = model_info['name']
    
    # Filter to niche
    df_niche = df[df['niche'].astype(str).str.upper() == niche.upper()].copy()
    
    if len(df_niche) < 20:
        print(f"⚠️ Not enough data for shuffled test: {len(df_niche)} rows")
        return None
    
    # Build features
    X, y = build_feature_matrix(df_niche, encoder)
    
    # Align with model features if needed
    if hasattr(model, 'feature_names_in_'):
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Shuffle labels
    y_train_shuffled = y_train.copy()
    np.random.seed(42)
    np.random.shuffle(y_train_shuffled)
    
    print(f"Training on {len(X_train)} samples with SHUFFLED labels...")
    
    # Train temporary model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        temp_model = RandomForestClassifier(
            n_estimators=50,  # Smaller for speed
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        temp_model.fit(X_train, y_train_shuffled)
    
    # Evaluate on REAL test labels
    y_pred = temp_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nShuffled-label model accuracy: {accuracy:.4f}")
    print(f"Expected: ~0.50 (random chance)")
    
    if accuracy > 0.55:
        print("⚠️ Overfitting detected — model learns noise!")
        return False
    elif accuracy < 0.45:
        print("⚠️ Unexpectedly low accuracy (possible bug)")
        return False
    else:
        print("✔️ Model behaves correctly on shuffled labels")
        return True


def test_prediction_distribution(model_info, test_results):
    """Test 6: Distribution of predictions."""
    print_section(f"TEST 6: PREDICTION DISTRIBUTION - {model_info['name']}")
    
    if test_results is None:
        print("⚠️ Skipping - no test results available")
        return False
    
    y_pred = test_results['y_pred']
    
    unique, counts = np.unique(y_pred, return_counts=True)
    total = len(y_pred)
    
    print("Prediction distribution:")
    for label, count in zip(unique, counts):
        pct = (count / total) * 100
        label_name = "High" if label == 1 else "Low"
        print(f"  {label_name}: {count} ({pct:.1f}%)")
    
    # Check for imbalance
    max_pct = max(counts) / total
    if max_pct > 0.90:
        print(f"⚠️ Highly imbalanced predictions ({max_pct*100:.1f}% one class)")
        return False
    else:
        print("✔️ Predictions are reasonably balanced")
        return True


def run_all_tests(model_info, df):
    """Run all tests for a model."""
    print("\n" + "=" * 60)
    print(f" TESTING {model_info['name']} MODEL")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Standard metrics
    test_results = test_standard_metrics(model_info, df)
    results['standard_metrics'] = test_results
    
    # Test 2: Class collapse
    results['class_collapse'] = test_class_collapse(model_info, test_results)
    
    # Test 3: Probability sanity
    results['probability_sanity'] = test_probability_sanity(model_info, test_results)
    
    # Test 4: Zero features
    results['zero_features'] = test_zero_features(model_info, model_info['encoder'])
    
    # Test 5: Shuffled labels
    results['shuffled_labels'] = test_shuffled_labels(model_info, df)
    
    # Test 6: Distribution
    results['distribution'] = test_prediction_distribution(model_info, test_results)
    
    return results


def final_verdict(all_results):
    """Print final verdict."""
    print_section("FINAL VERDICT")
    
    passed_tests = 0
    total_tests = 0
    
    for niche, results in all_results.items():
        if results is None:
            continue
        
        print(f"\n{niche.upper()} Model:")
        for test_name, result in results.items():
            total_tests += 1
            if result is True:
                passed_tests += 1
                print(f"  ✔️ {test_name}")
            elif result is False:
                print(f"  ❌ {test_name}")
            else:
                print(f"  ⚠️ {test_name} (skipped)")
    
    print(f"\n{'=' * 60}")
    print(f"Summary: {passed_tests}/{total_tests} tests passed")
    print(f"{'=' * 60}\n")
    
    if passed_tests == total_tests:
        print("✅ PASS → Model is learning real patterns")
    elif passed_tests >= total_tests * 0.7:
        print("⚠️ PARTIAL PASS → Some issues detected, review warnings")
    else:
        print("❌ FAIL → Model likely hallucinating or overfitting")


def main():
    """Main test runner."""
    print("=" * 60)
    print(" VIRALVISION MODEL INTEGRITY TEST SUITE")
    print("=" * 60)
    
    # Load models and dataset
    data = load_models_and_dataset()
    
    if data is None:
        print("\n❌ Cannot proceed without models and dataset")
        sys.exit(1)
    
    df = data['dataset']
    all_results = {}
    
    # Test GRWM model
    if data['grwm'] is not None:
        all_results['grwm'] = run_all_tests(data['grwm'], df)
    else:
        print("\n⚠️ Skipping GRWM tests (model not found)")
        all_results['grwm'] = None
    
    # Test OOTD model
    if data['ootd'] is not None:
        all_results['ootd'] = run_all_tests(data['ootd'], df)
    else:
        print("\n⚠️ Skipping OOTD tests (model not found)")
        all_results['ootd'] = None
    
    # Final verdict
    final_verdict(all_results)


if __name__ == "__main__":
    main()

