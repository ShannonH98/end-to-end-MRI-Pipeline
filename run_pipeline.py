from preprocessing.fsl_preprocessing import preprocess_folder
from preprocessing.slice_extraction import extract_slices
import os

RAW = "data/raw"
PROCESSED = "data/processed"
SLICES = "data/slices"

# Step 1 — Skull strip
print("=== Step 1: Skull Stripping ===")
preprocess_folder(RAW, PROCESSED)

# Step 2 — Extract slices per subject
print("\n=== Step 2: Slice Extraction ===")
for file in sorted(os.listdir(PROCESSED)):
    if file.endswith(".nii.gz"):
        subject = file.replace("_brain.nii.gz", "")
        path = os.path.join(PROCESSED, file)
        output_folder = os.path.join(SLICES, subject)
        extract_slices(path, output_folder)

print("\n=== Pipeline Complete ===")