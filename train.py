import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import defaultdict, Counter
from preprocessing.dataset import BrainSliceDataset
from preprocessing.label_map import load_labels
from models.cnn import BrainCNN
import json

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

# --- Datasets ---
train_dataset = BrainSliceDataset(SLICES, TSV, subject_ids=set(train_subs), include_augmented=True)
val_dataset   = BrainSliceDataset(SLICES, TSV, subject_ids=set(val_subs),   include_augmented=False)
test_dataset  = BrainSliceDataset(SLICES, TSV, subject_ids=set(test_subs),  include_augmented=False)

original_slices = sum(1 for path, _, _ in train_dataset.samples if "_aug" not in path)
augmented_slices = len(train_dataset) - original_slices
print(f"Train slices — original: {original_slices}, augmented: {augmented_slices}, total: {len(train_dataset)}")
print(f"Val slices: {len(val_dataset)} | Test slices: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

# --- Model ---
model = BrainCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

history = {
    "loss": [], "val_accuracy": [], "val_labels": [], "val_preds": [],
    "train_subjects": train_subs, "val_subjects": val_subs, "test_subjects": test_subs,
    "original_slices": original_slices, "augmented_slices": augmented_slices
}

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

# --- Training loop ---
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

    val_labels, val_preds, val_acc = evaluate(model, val_loader)
    history["loss"].append(total_loss)
    history["val_accuracy"].append(val_acc)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Val Subject Acc: {val_acc:.2f}%")

history["val_labels"] = val_labels
history["val_preds"] = val_preds

# --- Test set evaluation ---
test_labels, test_preds, test_acc = evaluate(model, test_loader)
print(f"\nTest Subject Accuracy: {test_acc:.2f}%")
print("\nVal per-class report:")
print(classification_report(val_labels, val_preds, target_names=CLASS_NAMES))
print("Test per-class report:")
print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))

# --- Save ---
torch.save(model.state_dict(), "models/brain_cnn.pth")
with open("models/history.json", "w") as f:
    json.dump(history, f)

print("\nModel saved → models/brain_cnn.pth")
print("Metrics saved → models/history.json")
