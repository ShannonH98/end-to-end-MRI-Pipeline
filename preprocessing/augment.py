import os
import json
from datetime import datetime
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as TF

SLICES = "data/slices"
LOG_FILE = "augment_log.json"

def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            return json.load(f)
    return []

def save_log(log):
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

def augment_subject(subject_folder, log):
    existing = os.listdir(subject_folder)
    subject = os.path.basename(subject_folder)

    if any("_aug" in f for f in existing):
        print(f"Skipping {subject} — already augmented")
        log["skipped"].append(subject)
        return

    originals = [f for f in sorted(existing) if f.endswith(".png") and "_aug" not in f]

    for fname in originals:
        img = Image.open(os.path.join(subject_folder, fname)).convert("L")
        stem = fname.replace(".png", "")

        TF.hflip(img).save(os.path.join(subject_folder, f"{stem}_aug1.png"))
        TF.rotate(img, angle=10).save(os.path.join(subject_folder, f"{stem}_aug2.png"))
        ImageEnhance.Brightness(img).enhance(1.2).save(os.path.join(subject_folder, f"{stem}_aug3.png"))

    new_slices = len(originals) * 3
    print(f"Augmented {subject} — {new_slices} new slices")
    log["augmented"].append({"subject": subject, "new_slices": new_slices})

if __name__ == "__main__":
    pipeline_log = load_log()

    run_entry = {
        "timestamp": datetime.now().isoformat(),
        "augmentation": {
            "techniques": ["horizontal_flip", "rotation_10deg", "brightness_1.2x"],
            "augmented": [],
            "skipped": []
        }
    }

    for subject in sorted(os.listdir(SLICES)):
        path = os.path.join(SLICES, subject)
        if os.path.isdir(path):
            augment_subject(path, run_entry["augmentation"])

    pipeline_log.append(run_entry)
    save_log(pipeline_log)

    print("\nAugmentation complete")
    print(f"Log saved to {LOG_FILE}")
