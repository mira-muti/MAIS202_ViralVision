import os
import shutil
import subprocess
import glob

ROOT = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_RAW = os.path.join(ROOT, "data/raw")
DATA_PROCESSED = os.path.join(ROOT, "data/processed")
FINAL_DATASET = os.path.join(ROOT, "data/processed/final_dataset.csv")
MODEL_PATH = os.path.join(ROOT, "models/model.pkl")
ENCODER_PATH = os.path.join(ROOT, "models/model_encoder.pkl")

SEARCH_DIRS = [
    ROOT,
    os.path.join(ROOT, "data"),
    os.path.join(ROOT, "data/raw"),
    os.path.join(ROOT, "data/processed"),
    os.path.join(ROOT, "models"),
]

def safe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[+] created {path}")

def create_placeholder(path, text):
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(text)
        print(f"[+] created placeholder: {path}")

def search_locally(pattern):
    """Search ONLY inside project folders. Not entire Mac."""
    for root_dir in SEARCH_DIRS:
        matches = glob.glob(os.path.join(root_dir, pattern))
        if matches:
            return matches[0]
    return None

def restore_if_missing(target_path, pattern, label):
    if os.path.exists(target_path):
        print(f"[✓] {label} already exists.")
        return True

    found = search_locally(pattern)
    if found:
        print(f"[+] Restoring {label} from {found}")
        shutil.copy(found, target_path)
        return True

    print(f"[!] {label} not found.")
    return False

def fix_5_second_extractor():
    extractor = os.path.join(ROOT, "src/extract_audio_features.py")
    if not os.path.exists(extractor):
        print("[!] extractor missing.")
        return

    with open(extractor, "r") as f:
        code = f.read()

    if "min(5" in code:
        print("[✓] Extractor already patched.")
        return

    patched = code.replace("min(seconds, clip.duration)", "min(5, clip.duration)")
    with open(extractor, "w") as f:
        f.write(patched)

    print("[+] Patched audio extractor for 5-second clips.")

def retrain_if_needed():
    missing = not (os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH))

    if not os.path.exists(FINAL_DATASET):
        print("[!] No final_dataset.csv — cannot train.")
        return

    if missing:
        print("[+] Retraining model...")
        subprocess.run(["python3", "src/train_model.py"], check=True)
        print("[✓] Model retrained.")
    else:
        print("[✓] Models exist — skipping retrain.")

def main():
    print("================================================")
    print("   SAFE RESTORE + FIX STARTING")
    print("================================================")

    safe_make_dir(DATA_RAW)
    safe_make_dir(DATA_PROCESSED)
    safe_make_dir(os.path.join(ROOT, "models"))

    create_placeholder(os.path.join(DATA_RAW, "README.md"), "placeholder")
    create_placeholder(os.path.join(DATA_PROCESSED, "README.md"), "placeholder")

    restore_if_missing(FINAL_DATASET, "final_dataset.csv", "Final dataset")
    restore_if_missing(MODEL_PATH, "model.pkl", "Random Forest model")
    restore_if_missing(ENCODER_PATH, "model_encoder.pkl", "Encoder")

    fix_5_second_extractor()
    retrain_if_needed()

    print("================================================")
    print("   RESTORE + FIX COMPLETE")
    print("================================================")

if __name__ == "__main__":
    main()