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
from preprocessing.label_map import load_labels
from models.cnn import BrainCNN
from models.BrainResNet import BrainResNet
from models.BrainRadImageNet import BrainRadImageNet

# Config
SLICES = "data/slices"
TSV = "data/participants.tsv"
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
N_FOLDS = 5
SEED = 42
CLASS_NAMES = ["HC", "AVH-", "AVH+"]
RADIMAGENET_WEIGHTS = "models/RadImageNet_resnet50.pt"

labels = load_labels(TSV)
all_subjects = list(labels.keys())
all_groups = [labels[s] for s in all_subjects]

use_radimagenet = os.path.exists(RADIMAGENET_WEIGHTS)
if not use_radimagenet:
    print("RadImageNet weights not found — skipping RadImageNet model.")
    print(f"Download weights and save to: {RADIMAGENET_WEIGHTS}")

print(f"Total subjects: {len(all_subjects)} | Running {N_FOLDS}-fold cross-validation")


class BrainDataset(Dataset):
    def __init__(self, slices_folder, tsv_path, image_size, grayscale_channels, subject_ids, include_augmented):
        self.samples = []
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ) if grayscale_channels == 3 else transforms.Normalize(mean=[0.5], std=[0.5])

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=grayscale_channels),
            transforms.ToTensor(),
            normalize,
        ])

        labels = load_labels(tsv_path)

        for subject in os.listdir(slices_folder):
            subject_path = os.path.join(slices_folder, subject)
            if not os.path.isdir(subject_path):
                continue
            pid = "sub-" + subject.split("-")[1].split("_")[0]
            if pid not in labels or pid not in subject_ids:
                continue

            orig_slices = sorted([f for f in os.listdir(subject_path) if f.endswith(".png") and "_aug" not in f])
            n = len(orig_slices)
            start, end = int(0.25 * n), int(0.75 * n)
            selected = set(orig_slices[start:end])

            for slice_file in sorted(os.listdir(subject_path)):
                if not slice_file.endswith(".png"):
                    continue
                if not include_augmented and "_aug" in slice_file:
                    continue
                base = slice_file.split("_aug")[0] + ".png" if "_aug" in slice_file else slice_file
                if base not in selected:
                    continue
                self.samples.append((os.path.join(subject_path, slice_file), labels[pid], pid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, subject_id = self.samples[idx]
        img = Image.open(path).convert("L")
        img = self.transform(img)
        return img, label, subject_id


def make_loaders(train_subs, test_subs, image_size, grayscale_channels):
    train_ds = BrainDataset(SLICES, TSV, image_size, grayscale_channels, set(train_subs), include_augmented=True)
    test_ds  = BrainDataset(SLICES, TSV, image_size, grayscale_channels, set(test_subs),  include_augmented=False)
    print(f"  Train slices: {len(train_ds)} | Test: {len(test_ds)}")
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(test_ds,  batch_size=BATCH_SIZE),
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
    all_preds = [Counter(subject_preds[sid]).most_common(1)[0][0] for sid in subject_labels]
    acc = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return all_labels, all_preds, acc


def train_model(model, train_loader, test_loader, optimizer, name, class_weights=None):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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
        print(f"  [{name}] Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Test Acc: {acc:.2f}%")


# --- K-Fold Cross-Validation ---
kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

cnn_fold_accs       = []
resnet_fold_accs    = []
radimagenet_fold_accs = []
cnn_fold_results       = []
resnet_fold_results    = []
radimagenet_fold_results = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(all_subjects, all_groups)):
    train_subs = [all_subjects[i] for i in train_idx]
    test_subs  = [all_subjects[i] for i in test_idx]

    print(f"\n{'='*55}")
    print(f"FOLD {fold+1}/{N_FOLDS} — Train: {len(train_subs)} subjects | Test: {len(test_subs)} subjects")
    print(f"{'='*55}")

    # Class weights from training fold
    fold_counts = Counter([all_groups[i] for i in train_idx])
    class_weights = torch.tensor(
        [len(train_subs) / (3 * fold_counts[c]) for c in range(3)], dtype=torch.float
    )
    print(f"  Class weights: HC={class_weights[0]:.2f}, AVH-={class_weights[1]:.2f}, AVH+={class_weights[2]:.2f}")

    # --- CNN ---
    print("\n--- BrainCNN ---")
    cnn_train_loader, cnn_test_loader = make_loaders(train_subs, test_subs, 128, 1)
    cnn_model = BrainCNN()
    cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
    train_model(cnn_model, cnn_train_loader, cnn_test_loader, cnn_optimizer, "CNN", class_weights)
    cnn_labels, cnn_preds, cnn_acc = evaluate(cnn_model, cnn_test_loader)
    cnn_fold_accs.append(cnn_acc)
    cnn_fold_results.append({"fold": fold+1, "test_labels": cnn_labels, "test_preds": cnn_preds, "acc": cnn_acc})
    print(f"  CNN Fold {fold+1} Test Acc: {cnn_acc:.2f}%")

    # --- ResNet ---
    print("\n--- BrainResNet ---")
    resnet_train_loader, resnet_test_loader = make_loaders(train_subs, test_subs, 224, 3)
    resnet_model = BrainResNet()
    resnet_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, resnet_model.parameters()), lr=LEARNING_RATE)
    train_model(resnet_model, resnet_train_loader, resnet_test_loader, resnet_optimizer, "ResNet", class_weights)
    resnet_labels, resnet_preds, resnet_acc = evaluate(resnet_model, resnet_test_loader)
    resnet_fold_accs.append(resnet_acc)
    resnet_fold_results.append({"fold": fold+1, "test_labels": resnet_labels, "test_preds": resnet_preds, "acc": resnet_acc})
    print(f"  ResNet Fold {fold+1} Test Acc: {resnet_acc:.2f}%")

    # --- RadImageNet ---
    if use_radimagenet:
        print("\n--- BrainRadImageNet ---")
        rad_train_loader, rad_test_loader = make_loaders(train_subs, test_subs, 224, 3)
        rad_model = BrainRadImageNet(weights_path=RADIMAGENET_WEIGHTS)
        rad_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, rad_model.parameters()), lr=LEARNING_RATE)
        train_model(rad_model, rad_train_loader, rad_test_loader, rad_optimizer, "RadImageNet", class_weights)
        rad_labels, rad_preds, rad_acc = evaluate(rad_model, rad_test_loader)
        radimagenet_fold_accs.append(rad_acc)
        radimagenet_fold_results.append({"fold": fold+1, "test_labels": rad_labels, "test_preds": rad_preds, "acc": rad_acc})
        print(f"  RadImageNet Fold {fold+1} Test Acc: {rad_acc:.2f}%")

# --- Summary ---
print("\n" + "="*65)
print("K-FOLD SUMMARY (subject-level majority vote)")
print("="*65)
header = f"\n{'Fold':<20} {'BrainCNN':>14} {'BrainResNet':>14}"
if use_radimagenet:
    header += f" {'RadImageNet':>14}"
print(header)
print("-"*65)
for i, (ca, ra) in enumerate(zip(cnn_fold_accs, resnet_fold_accs)):
    row = f"  Fold {i+1:<15} {ca:>13.2f}% {ra:>13.2f}%"
    if use_radimagenet:
        row += f" {radimagenet_fold_accs[i]:>13.2f}%"
    print(row)
print("-"*65)
mean_row = f"{'Mean Accuracy':<20} {np.mean(cnn_fold_accs):>13.2f}% {np.mean(resnet_fold_accs):>13.2f}%"
std_row  = f"{'Std Dev':<20} {np.std(cnn_fold_accs):>13.2f}% {np.std(resnet_fold_accs):>13.2f}%"
if use_radimagenet:
    mean_row += f" {np.mean(radimagenet_fold_accs):>13.2f}%"
    std_row  += f" {np.std(radimagenet_fold_accs):>13.2f}%"
print(mean_row)
print(std_row)
print(f"{'Chance baseline':<20} {'33.33%':>14} {'33.33%':>14}")

# --- Save ---
results = {
    "n_folds": N_FOLDS, "epochs": EPOCHS, "seed": SEED,
    "cnn":    {"fold_accs": cnn_fold_accs,    "mean": np.mean(cnn_fold_accs),    "std": np.std(cnn_fold_accs),    "folds": cnn_fold_results},
    "resnet": {"fold_accs": resnet_fold_accs, "mean": np.mean(resnet_fold_accs), "std": np.std(resnet_fold_accs), "folds": resnet_fold_results},
}
if use_radimagenet:
    results["radimagenet"] = {
        "fold_accs": radimagenet_fold_accs,
        "mean": np.mean(radimagenet_fold_accs),
        "std": np.std(radimagenet_fold_accs),
        "folds": radimagenet_fold_results,
    }
with open("models/kfold_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved → models/kfold_results.json")
