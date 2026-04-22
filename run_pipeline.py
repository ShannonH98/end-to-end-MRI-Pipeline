from preprocessing.fsl_preprocessing import preprocess_folder
from preprocessing.slice_extraction import extract_slices
import os
import json
from datetime import datetime

RAW = "data/raw"
PROCESSED = "data/processed"
SLICES = "data/slices"
LOG_FILE = "pipeline_log.json"

def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            return json.load(f)
    return []

def save_log(log):
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

log = load_log()
run_entry = {
    "timestamp": datetime.now().isoformat(),
    "skull_stripping": {
        "parameters": {"f": 0.5, "g": 0},
        "processed": [],
        "skipped": []
    },
    "slice_extraction": {
        "empty_slice_threshold": 10,
        "normalization": "min-max to 0-255",
        "processed": [],
        "skipped": []
    }
}

# Step 1 — Skull strip
print("=== Step 1: Skull Stripping ===")
preprocess_folder(RAW, PROCESSED, run_entry["skull_stripping"])

# Step 2 — Extract slices per subject
print("\n=== Step 2: Slice Extraction ===")
for file in sorted(os.listdir(PROCESSED)):
    if file.endswith(".nii.gz"):
        subject = file.replace("_brain.nii.gz", "")
        path = os.path.join(PROCESSED, file)
        output_folder = os.path.join(SLICES, subject)
        extract_slices(path, output_folder, run_entry["slice_extraction"])

log.append(run_entry)
save_log(log)

print("\n=== Pipeline Complete ===")
print(f"Log saved to {LOG_FILE}")
