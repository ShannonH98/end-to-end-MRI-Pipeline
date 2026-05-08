import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict, Counter
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from preprocessing.label_map import load_labels
from models.cnn import BrainCNN
from models.BrainResNet import BrainResNet

# Config
SLICES = "data/slices"
TSV = "data/participants.tsv"
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
SEED = 42
CLASS_NAMES = ["HC", "AVH-", "AVH+"]

# --- Subject-level split ---
labels = load_labels(TSV)
all_subjects = list(labels.keys())
all_groups = [labels[s] for s in all_subjects]

train_subs, temp_subs, train_groups, temp_groups = train_test_split(
    all_subjects, all_groups, test_size=0.3, random_state=SEED, stratify=all_groups
)
val_subs, test_subs = train_test_split(
    temp_subs, test_size=0.5, random_state=SEED, stratify=temp_groups
)

print(f"Subjects — train: {len(train_subs)}, val: {len(val_subs)}, test: {len(test_subs)}")

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
            for slice_file in sorted(os.listdir(subject_path)):
                if slice_file.endswith(".png"):
                    if not include_augmented and "_aug" in slice_file:
                        continue
                    self.samples.append((os.path.join(subject_path, slice_file), labels[pid], pid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, subject_id = self.samples[idx]
        img = Image.open(path).convert("L")
        img = self.transform(img)
        return img, label, subject_id


def make_loaders(image_size, grayscale_channels):
    train_ds = BrainDataset(SLICES, TSV, image_size, grayscale_channels, set(train_subs), include_augmented=True)
    val_ds   = BrainDataset(SLICES, TSV, image_size, grayscale_channels, set(val_subs),   include_augmented=False)
    test_ds  = BrainDataset(SLICES, TSV, image_size, grayscale_channels, set(test_subs),  include_augmented=False)
    print(f"  Train slices: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_ds,   batch_size=BATCH_SIZE),
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


def train_model(model, train_loader, val_loader, optimizer, name):
    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "val_accuracy": []}
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
        _, _, acc = evaluate(model, val_loader)
        history["loss"].append(total_loss)
        history["val_accuracy"].append(acc)
        print(f"  [{name}] Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Val Acc: {acc:.2f}%")
    return history


# --- CNN ---
print("\n=== Training BrainCNN ===")
cnn_train_loader, cnn_val_loader, cnn_test_loader = make_loaders(128, 1)
cnn_model = BrainCNN()
cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
cnn_history = train_model(cnn_model, cnn_train_loader, cnn_val_loader, cnn_optimizer, "CNN")
cnn_val_labels, cnn_val_preds, _ = evaluate(cnn_model, cnn_val_loader)
cnn_test_labels, cnn_test_preds, cnn_test_acc = evaluate(cnn_model, cnn_test_loader)

# --- ResNet ---
print("\n=== Training BrainResNet ===")
resnet_train_loader, resnet_val_loader, resnet_test_loader = make_loaders(224, 3)
resnet_model = BrainResNet()
resnet_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, resnet_model.parameters()), lr=LEARNING_RATE)
resnet_history = train_model(resnet_model, resnet_train_loader, resnet_val_loader, resnet_optimizer, "ResNet")
resnet_val_labels, resnet_val_preds, _ = evaluate(resnet_model, resnet_val_loader)
resnet_test_labels, resnet_test_preds, resnet_test_acc = evaluate(resnet_model, resnet_test_loader)

# --- Comparison ---
print("\n" + "="*55)
print("MODEL COMPARISON (subject-level, no data leakage)")
print("="*55)
print(f"\n{'Metric':<20} {'BrainCNN':>15} {'BrainResNet':>15}")
print("-"*55)
print(f"{'Best Val Acc':<20} {max(cnn_history['val_accuracy']):>14.2f}% {max(resnet_history['val_accuracy']):>14.2f}%")
print(f"{'Test Acc':<20} {cnn_test_acc:>14.2f}% {resnet_test_acc:>14.2f}%")

print("\n--- BrainCNN test report ---")
print(classification_report(cnn_test_labels, cnn_test_preds, target_names=CLASS_NAMES))
print("--- BrainResNet test report ---")
print(classification_report(resnet_test_labels, resnet_test_preds, target_names=CLASS_NAMES))

# --- Save ---
results = {
    "seed": SEED, "epochs": EPOCHS,
    "subjects": {"train": train_subs, "val": val_subs, "test": test_subs},
    "cnn": {"history": cnn_history, "test_labels": cnn_test_labels, "test_preds": cnn_test_preds},
    "resnet": {"history": resnet_history, "test_labels": resnet_test_labels, "test_preds": resnet_test_preds}
}
with open("models/comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)

torch.save(cnn_model.state_dict(), "models/brain_cnn.pth")
torch.save(resnet_model.state_dict(), "models/brain_resnet.pth")
print("\nResults saved → models/comparison_results.json")
