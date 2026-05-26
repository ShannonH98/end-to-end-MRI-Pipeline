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
# Cap per class to keep training time manageable and the dataset balanced.
# Non Demented has 67k images; without a cap training takes days on CPU.
MAX_IMAGES_PER_CLASS = 3000

CLASS_NAMES = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]
LABEL_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

use_radimagenet = os.path.exists(RADIMAGENET_WEIGHTS)
if not use_radimagenet:
    print("RadImageNet weights not found — skipping RadImageNet model.")

rng = np.random.default_rng(SEED)


def build_subject_index(oasis_dir):
    """
    Returns list of (subject_id, class_label, [image_paths]).
    - Subject ID extracted from filename: OAS1_0001 from OAS1_0001_MR1_mpr-1_100.jpg
    - Only the middle 50% of slices per subject are kept (matches original pipeline).
    - Each class is capped at MAX_IMAGES_PER_CLASS total images to keep training feasible.
    """
    subject_map = {}
    for class_name, label in LABEL_MAP.items():
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
    trimmed = {}
    for (sid, lbl), paths in subject_map.items():
        paths = sorted(paths)
        n = len(paths)
        start, end = int(0.25 * n), int(0.75 * n)
        trimmed[(sid, lbl)] = paths[start:end]

    # Cap each class to MAX_IMAGES_PER_CLASS by randomly dropping subjects
    by_class = defaultdict(list)
    for (sid, lbl), paths in trimmed.items():
        by_class[lbl].append((sid, lbl, paths))

    subjects = []
    for lbl, entries in by_class.items():
        total = sum(len(p) for _, _, p in entries)
        if total > MAX_IMAGES_PER_CLASS:
            # Shuffle and accumulate subjects until we hit the cap
            idx = rng.permutation(len(entries)).tolist()
            kept, count = [], 0
            for i in idx:
                sid, lbl_, paths = entries[i]
                kept.append((sid, lbl_, paths))
                count += len(paths)
                if count >= MAX_IMAGES_PER_CLASS:
                    break
            subjects.extend(kept)
        else:
            subjects.extend(entries)

    return subjects


class OasisDataset(Dataset):
    def __init__(self, subject_entries, image_size, grayscale_channels):
        self.samples = []
        normalize = (
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if grayscale_channels == 3
            else transforms.Normalize(mean=[0.5], std=[0.5])
        )
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=grayscale_channels),
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
    train_ds = OasisDataset(train_entries, image_size, grayscale_channels)
    test_ds  = OasisDataset(test_entries,  image_size, grayscale_channels)
    print(f"  Train slices: {len(train_ds)} | Test: {len(test_ds)}")
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0),
        DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    )


def evaluate(model, loader):
    model.eval()
    subject_preds = defaultdict(list)
    subject_labels = {}
    with torch.no_grad():
        for images, labels, subject_ids in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for sid, pred, lbl in zip(subject_ids, predicted.tolist(), labels.tolist()):
                subject_preds[sid].append(pred)
                subject_labels[sid] = lbl
    all_labels = list(subject_labels.values())
    all_preds  = [Counter(subject_preds[sid]).most_common(1)[0][0] for sid in subject_labels]
    acc = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return all_labels, all_preds, acc


def train_model(model, train_loader, test_loader, optimizer, name, class_weights=None):
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
        _, _, acc = evaluate(model, test_loader)
        history["loss"].append(total_loss)
        history["accuracy"].append(acc)
        print(f"  [{name}] Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Test Acc: {acc:.2f}%")
    return history


# --- Build subject index ---
all_subjects = build_subject_index(OASIS_DIR)
all_labels_for_split = [lbl for _, lbl, _ in all_subjects]

print(f"OASIS subjects: {len(all_subjects)}")
for i, name in enumerate(CLASS_NAMES):
    count = all_labels_for_split.count(i)
    print(f"  {name}: {count} subjects")

print(f"\nRunning {N_FOLDS}-fold cross-validation\n")

# --- K-Fold ---
kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

cnn_fold_accs, cnn_fold_results             = [], []
resnet_fold_accs, resnet_fold_results       = [], []
radimagenet_fold_accs, radimagenet_fold_results = [], []

for fold, (train_idx, test_idx) in enumerate(kfold.split(all_subjects, all_labels_for_split)):
    train_entries = [all_subjects[i] for i in train_idx]
    test_entries  = [all_subjects[i] for i in test_idx]

    print(f"\n{'='*60}")
    print(f"FOLD {fold+1}/{N_FOLDS} — Train: {len(train_entries)} subjects | Test: {len(test_entries)} subjects")
    print(f"{'='*60}")

    fold_counts = Counter([all_labels_for_split[i] for i in train_idx])
    n_train = len(train_entries)
    class_weights = torch.tensor(
        [n_train / (4 * max(fold_counts[c], 1)) for c in range(4)], dtype=torch.float
    )
    print(f"  Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

    # --- OasisCNN ---
    print("\n--- OasisCNN ---")
    cnn_train, cnn_test = make_loaders(train_entries, test_entries, 128, 1)
    cnn_model = OasisCNN(num_classes=4)
    cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
    cnn_history = train_model(cnn_model, cnn_train, cnn_test, cnn_optimizer, "OasisCNN", class_weights)
    cnn_labels, cnn_preds, cnn_acc = evaluate(cnn_model, cnn_test)
    cnn_fold_accs.append(cnn_acc)
    cnn_fold_results.append({"fold": fold+1, "test_labels": cnn_labels, "test_preds": cnn_preds, "acc": cnn_acc, "history": cnn_history})
    print(f"  OasisCNN Fold {fold+1} Test Acc: {cnn_acc:.2f}%")

    # --- ResNet ---
    print("\n--- BrainResNet (OASIS) ---")
    resnet_train, resnet_test = make_loaders(train_entries, test_entries, 224, 3)
    resnet_model = BrainResNet(num_classes=4)
    resnet_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, resnet_model.parameters()), lr=LEARNING_RATE
    )
    resnet_history = train_model(resnet_model, resnet_train, resnet_test, resnet_optimizer, "ResNet", class_weights)
    resnet_labels, resnet_preds, resnet_acc = evaluate(resnet_model, resnet_test)
    resnet_fold_accs.append(resnet_acc)
    resnet_fold_results.append({"fold": fold+1, "test_labels": resnet_labels, "test_preds": resnet_preds, "acc": resnet_acc, "history": resnet_history})
    print(f"  ResNet Fold {fold+1} Test Acc: {resnet_acc:.2f}%")

    # --- RadImageNet ---
    if use_radimagenet:
        print("\n--- BrainRadImageNet (OASIS) ---")
        rad_train, rad_test = make_loaders(train_entries, test_entries, 224, 3)
        rad_model = BrainRadImageNet(weights_path=RADIMAGENET_WEIGHTS, num_classes=4)
        rad_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, rad_model.parameters()), lr=LEARNING_RATE
        )
        rad_history = train_model(rad_model, rad_train, rad_test, rad_optimizer, "RadImageNet", class_weights)
        rad_labels, rad_preds, rad_acc = evaluate(rad_model, rad_test)
        radimagenet_fold_accs.append(rad_acc)
        radimagenet_fold_results.append({"fold": fold+1, "test_labels": rad_labels, "test_preds": rad_preds, "acc": rad_acc, "history": rad_history})
        print(f"  RadImageNet Fold {fold+1} Test Acc: {rad_acc:.2f}%")

# --- Summary ---
print("\n" + "="*70)
print("OASIS K-FOLD SUMMARY (subject-level majority vote)")
print("="*70)
header = f"\n{'Fold':<20} {'OasisCNN':>14} {'BrainResNet':>14}"
if use_radimagenet:
    header += f" {'RadImageNet':>14}"
print(header)
print("-"*70)
for i, (ca, ra) in enumerate(zip(cnn_fold_accs, resnet_fold_accs)):
    row = f"  Fold {i+1:<15} {ca:>13.2f}% {ra:>13.2f}%"
    if use_radimagenet:
        row += f" {radimagenet_fold_accs[i]:>13.2f}%"
    print(row)
print("-"*70)
mean_row = f"{'Mean Accuracy':<20} {np.mean(cnn_fold_accs):>13.2f}% {np.mean(resnet_fold_accs):>13.2f}%"
std_row  = f"{'Std Dev':<20} {np.std(cnn_fold_accs):>13.2f}% {np.std(resnet_fold_accs):>13.2f}%"
if use_radimagenet:
    mean_row += f" {np.mean(radimagenet_fold_accs):>13.2f}%"
    std_row  += f" {np.std(radimagenet_fold_accs):>13.2f}%"
print(mean_row)
print(std_row)
print(f"{'Chance baseline':<20} {'25.00%':>14} {'25.00%':>14}")

# --- Save ---
results = {
    "dataset": "OASIS",
    "classes": CLASS_NAMES,
    "n_folds": N_FOLDS,
    "epochs": EPOCHS,
    "seed": SEED,
    "oasis_cnn":  {"fold_accs": cnn_fold_accs,  "mean": np.mean(cnn_fold_accs),  "std": np.std(cnn_fold_accs),  "folds": cnn_fold_results},
    "resnet":     {"fold_accs": resnet_fold_accs, "mean": np.mean(resnet_fold_accs), "std": np.std(resnet_fold_accs), "folds": resnet_fold_results},
}
if use_radimagenet:
    results["radimagenet"] = {
        "fold_accs": radimagenet_fold_accs,
        "mean": np.mean(radimagenet_fold_accs),
        "std": np.std(radimagenet_fold_accs),
        "folds": radimagenet_fold_results,
    }

os.makedirs("models", exist_ok=True)
with open("models/oasis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved → models/oasis_results.json")
