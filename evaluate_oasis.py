import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict, Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from models.OasisCNN import OasisCNN
from models.BrainResNet import BrainResNet
from models.BrainRadImageNet import BrainRadImageNet

# Config
OASIS_DIR = "OASIS Data"
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
N_FOLDS = 5
SEED = 42
RADIMAGENET_WEIGHTS = "models/resnet50_torch.pt"

# FIX (item 1): No more subject cap. We use EVERY subject in each class.
# The old MAX_SUBJECTS_PER_CLASS=24 threw away ~242 of the 266 Non Demented
# subjects — an artifact of the original per-slice cap. Removed entirely.

# Two experiments are run back-to-back and saved to separate files so they
# can be compared directly:
#   - binary     : Non Demented (0) vs Demented (1).  Demented = Very Mild + Mild + Moderate.
#   - multiclass : Non Demented (0) / Very mild (1) / Mild (2).
#                  Moderate Dementia (only 2 subjects, never lands in a test fold)
#                  is MERGED into Mild so the multi-class task isn't silently broken.
MODE_CONFIGS = {
    "binary": {
        "num_classes": 2,
        "class_names": ["Non Demented", "Demented"],
        "label_map": {
            "Non Demented":       0,
            "Very mild Dementia": 1,
            "Mild Dementia":      1,
            "Moderate Dementia":  1,
        },
        "label_mapping_desc": "Non Demented=0, Very Mild+Mild+Moderate=1",
        "chance_baseline": 50.0,
        "out_path": "models/oasis_results_binary.json",
    },
    "multiclass": {
        "num_classes": 3,
        "class_names": ["Non Demented", "Very mild Dementia", "Mild Dementia"],
        "label_map": {
            "Non Demented":       0,
            "Very mild Dementia": 1,
            "Mild Dementia":      2,
            "Moderate Dementia":  2,  # merged into Mild (n=2 subjects only)
        },
        "label_mapping_desc": "Non Demented=0, Very mild=1, Mild+Moderate=2",
        "chance_baseline": 100.0 / 3,
        "out_path": "models/oasis_results_multiclass.json",
    },
}

use_radimagenet = os.path.exists(RADIMAGENET_WEIGHTS)
if not use_radimagenet:
    print("RadImageNet weights not found — skipping RadImageNet model.")


def bootstrap_ci_subjects(all_labels, all_preds, n_boot=2000, ci=95, seed=42):
    """
    Subject-level bootstrap CI.
    Pools every subject's (true_label, predicted_label) pair across all folds,
    then resamples subjects (not fold accuracies) to get a reliable CI.
    Returns native Python floats so json.dump never writes null.
    """
    if len(all_labels) == 0:
        return [0.0, 0.0]
    rng_b = np.random.default_rng(seed)
    pairs = np.array(list(zip(all_labels, all_preds)))  # shape (n_subjects, 2)
    boots = []
    for _ in range(n_boot):
        sample = pairs[rng_b.integers(0, len(pairs), size=len(pairs))]
        acc = 100 * np.mean(sample[:, 0] == sample[:, 1])
        boots.append(acc)
    lo = float(np.percentile(boots, (100 - ci) / 2))
    hi = float(np.percentile(boots, 100 - (100 - ci) / 2))
    return [lo, hi]


def build_subject_index(oasis_dir, label_map):
    """
    Returns list of (subject_id, label, [image_paths]) for the given label map.
    - Subject ID extracted from filename: OAS1_0001 from OAS1_0001_MR1_mpr-1_100.jpg
    - Only the middle 50% of slices per subject are kept (matches original pipeline).
    - NO subject cap (item 1): every subject in every class is used.
    """
    subject_map = {}
    for class_name, label in label_map.items():
        class_dir = os.path.join(oasis_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith(".jpg"):
                continue
            subject_id = "_".join(fname.split("_")[:2])  # OAS1_XXXX
            key = (subject_id, label)
            subject_map.setdefault(key, []).append(os.path.join(class_dir, fname))

    # Keep middle 50% of slices per subject (same logic as original BrainDataset)
    subjects = []
    for (sid, lbl), paths in subject_map.items():
        paths = sorted(paths)
        n = len(paths)
        start, end = int(0.25 * n), int(0.75 * n)
        subjects.append((sid, lbl, paths[start:end]))

    return subjects


class OasisDataset(Dataset):
    def __init__(self, subject_entries, image_size, grayscale_channels, augment=False):
        self.samples = []
        normalize = (
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if grayscale_channels == 3
            else transforms.Normalize(mean=[0.5], std=[0.5])
        )
        # FIX (item 4): augmentation is applied to TRAINING data only.
        # Test loaders are built with augment=False so evaluation stays deterministic.
        aug = []
        if augment:
            aug = [
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=grayscale_channels),
            *aug,
            transforms.ToTensor(),
            normalize,
        ])
        for subject_id, label, paths in subject_entries:
            for path in paths:
                self.samples.append((path, label, subject_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, subject_id = self.samples[idx]
        img = Image.open(path).convert("L")
        img = self.transform(img)
        return img, label, subject_id


def make_loaders(train_entries, test_entries, image_size, grayscale_channels):
    train_ds = OasisDataset(train_entries, image_size, grayscale_channels, augment=True)
    test_ds  = OasisDataset(test_entries,  image_size, grayscale_channels, augment=False)
    print(f"  Train slices: {len(train_ds)} | Test slices: {len(test_ds)}")
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0),
        DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    )


def log_fold_subjects(fold, train_entries, test_entries, class_names):
    """Print how many subjects (not slices) each class has in train and test."""
    train_counts = Counter(lbl for _, lbl, _ in train_entries)
    test_counts  = Counter(lbl for _, lbl, _ in test_entries)
    print(f"\n  Fold {fold} subject counts:")
    for i, name in enumerate(class_names):
        print(f"    {name:<22} train={train_counts.get(i, 0):>3} subjects | test={test_counts.get(i, 0):>3} subjects")


def evaluate(model, loader, num_classes):
    """
    Subject-level prediction via SOFTMAX PROBABILITY AVERAGING (item 5).
    Each slice contributes its softmax distribution; we average them per subject
    and argmax the mean. This is more stable than majority vote because it uses
    the model's confidence, not just the hard slice label.
    """
    model.eval()
    subject_probs  = defaultdict(lambda: np.zeros(num_classes, dtype=np.float64))
    subject_counts = defaultdict(int)
    subject_labels = {}
    with torch.no_grad():
        for images, labels, subject_ids in loader:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            for sid, p, lbl in zip(subject_ids, probs, labels.tolist()):
                subject_probs[sid]  += p
                subject_counts[sid] += 1
                subject_labels[sid]  = lbl
    all_labels, all_preds = [], []
    for sid in subject_labels:
        avg = subject_probs[sid] / subject_counts[sid]
        all_labels.append(subject_labels[sid])
        all_preds.append(int(np.argmax(avg)))
    acc = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return all_labels, all_preds, acc


def train_model(model, train_loader, test_loader, optimizer, name, num_classes, class_weights=None):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    history = {"loss": [], "accuracy": []}
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, labels, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        _, _, acc = evaluate(model, test_loader, num_classes)
        history["loss"].append(total_loss)
        history["accuracy"].append(acc)
        print(f"  [{name}] Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Test Acc: {acc:.2f}%")
    return history


def confusion_for_fold(labels, preds, num_classes):
    """Per-fold confusion matrix (item 6), rows=true, cols=pred, as a plain list."""
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    return cm.tolist()


def run_experiment(mode):
    cfg = MODE_CONFIGS[mode]
    num_classes = cfg["num_classes"]
    class_names = cfg["class_names"]

    # Fresh RNG per experiment so subject ordering is reproducible per mode.
    rng = np.random.default_rng(SEED)

    print("\n" + "#" * 70)
    print(f"# EXPERIMENT: {mode.upper()}  ({num_classes} classes — {cfg['label_mapping_desc']})")
    print("#" * 70)

    all_subjects = build_subject_index(OASIS_DIR, cfg["label_map"])
    # Deterministic shuffle so fold assignment doesn't depend on filesystem order.
    perm = rng.permutation(len(all_subjects))
    all_subjects = [all_subjects[i] for i in perm]
    all_labels_for_split = [lbl for _, lbl, _ in all_subjects]

    print(f"OASIS subjects loaded: {len(all_subjects)}")
    for i, name in enumerate(class_names):
        print(f"  {name}: {all_labels_for_split.count(i)} subjects")
    print(f"\nRunning {N_FOLDS}-fold cross-validation\n")

    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    model_keys = ["oasis_cnn", "resnet"] + (["radimagenet"] if use_radimagenet else [])
    fold_accs    = {k: [] for k in model_keys}
    fold_results = {k: [] for k in model_keys}
    all_labels   = {k: [] for k in model_keys}
    all_preds    = {k: [] for k in model_keys}

    for fold, (train_idx, test_idx) in enumerate(kfold.split(all_subjects, all_labels_for_split)):
        train_entries = [all_subjects[i] for i in train_idx]
        test_entries  = [all_subjects[i] for i in test_idx]

        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{N_FOLDS} — Train: {len(train_entries)} subjects | Test: {len(test_entries)} subjects")
        print(f"{'='*60}")
        log_fold_subjects(fold + 1, train_entries, test_entries, class_names)

        fold_counts = Counter([all_labels_for_split[i] for i in train_idx])
        n_train = len(train_entries)
        class_weights = torch.tensor(
            [n_train / (num_classes * max(fold_counts[c], 1)) for c in range(num_classes)],
            dtype=torch.float,
        )
        print(f"\n  Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

        def record(key, model, test_loader, history):
            labels, preds, acc = evaluate(model, test_loader, num_classes)
            fold_accs[key].append(acc)
            fold_results[key].append({
                "fold": fold + 1,
                "test_labels": labels,
                "test_preds": preds,
                "acc": acc,
                "confusion_matrix": confusion_for_fold(labels, preds, num_classes),
                "history": history,
            })
            all_labels[key].extend(labels)
            all_preds[key].extend(preds)
            print(f"  Fold {fold+1} Test Acc: {acc:.2f}%")
            return acc

        # --- OasisCNN ---
        print("\n--- OasisCNN ---")
        cnn_train, cnn_test = make_loaders(train_entries, test_entries, 128, 1)
        cnn_model = OasisCNN(num_classes=num_classes)
        cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
        cnn_history = train_model(cnn_model, cnn_train, cnn_test, cnn_optimizer, "OasisCNN", num_classes, class_weights)
        record("oasis_cnn", cnn_model, cnn_test, cnn_history)

        # --- ResNet ---
        print("\n--- BrainResNet (OASIS) ---")
        resnet_train, resnet_test = make_loaders(train_entries, test_entries, 224, 3)
        resnet_model = BrainResNet(num_classes=num_classes)
        resnet_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, resnet_model.parameters()), lr=LEARNING_RATE
        )
        resnet_history = train_model(resnet_model, resnet_train, resnet_test, resnet_optimizer, "ResNet", num_classes, class_weights)
        record("resnet", resnet_model, resnet_test, resnet_history)

        # --- RadImageNet ---
        if use_radimagenet:
            print("\n--- BrainRadImageNet (OASIS) ---")
            rad_train, rad_test = make_loaders(train_entries, test_entries, 224, 3)
            rad_model = BrainRadImageNet(weights_path=RADIMAGENET_WEIGHTS, num_classes=num_classes)
            rad_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, rad_model.parameters()), lr=LEARNING_RATE
            )
            rad_history = train_model(rad_model, rad_train, rad_test, rad_optimizer, "RadImageNet", num_classes, class_weights)
            record("radimagenet", rad_model, rad_test, rad_history)

    # --- Summary ---
    label_for = {"oasis_cnn": "OasisCNN", "resnet": "BrainResNet", "radimagenet": "RadImageNet"}
    print("\n" + "=" * 70)
    print(f"OASIS K-FOLD SUMMARY — {mode.upper()} ({cfg['label_mapping_desc']})")
    print(f"subject-level softmax averaging | chance baseline = {cfg['chance_baseline']:.2f}%")
    print("=" * 70)
    for key in model_keys:
        print(f"  {label_for[key]:<14} fold accs: {[f'{a:.1f}' for a in fold_accs[key]]}")
    print("-" * 70)
    for key in model_keys:
        lo, hi = bootstrap_ci_subjects(all_labels[key], all_preds[key])
        print(f"  {label_for[key]:<14} mean={np.mean(fold_accs[key]):6.2f}%  std={np.std(fold_accs[key]):5.2f}  "
              f"95% CI [{lo:.2f}%, {hi:.2f}%]  (n={len(all_labels[key])} subjects)")
    print(f"\n  Chance baseline: {cfg['chance_baseline']:.2f}%")
    print("  NOTE: Report the CI, not just the mean — small n means wide CIs are honest.")

    # --- Save ---
    results = {
        "dataset": "OASIS",
        "mode": mode,
        "classes": class_names,
        "label_mapping": cfg["label_mapping_desc"],
        "n_folds": N_FOLDS,
        "epochs": EPOCHS,
        "seed": SEED,
        "subject_cap": None,  # item 1: cap removed, all subjects used
        "chance_baseline": cfg["chance_baseline"],
        "prediction_method": "softmax probability averaging",
        "augmentation": "train-only: RandomRotation(10), RandomAffine translate=0.05",
    }
    for key in model_keys:
        results[key] = {
            "fold_accs": fold_accs[key],
            "mean": float(np.mean(fold_accs[key])),
            "std": float(np.std(fold_accs[key])),
            "ci_95_subject_level": bootstrap_ci_subjects(all_labels[key], all_preds[key]),
            "n_subjects_pooled": len(all_labels[key]),
            "folds": fold_results[key],
        }

    os.makedirs("models", exist_ok=True)
    with open(cfg["out_path"], "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {cfg['out_path']}")
    return results


if __name__ == "__main__":
    for mode in ["binary", "multiclass"]:
        run_experiment(mode)
