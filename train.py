import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from preprocessing.dataset import BrainSliceDataset
from models.cnn import BrainCNN
from collections import defaultdict, Counter
from sklearn.metrics import classification_report
import json

# Config
SLICES = "data/slices"
TSV = "data/participants.tsv"
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Dataset
dataset = BrainSliceDataset(SLICES, TSV)

total_slices = len(dataset)
original_slices = sum(1 for path, _, _ in dataset.samples if "_aug" not in path)
augmented_slices = total_slices - original_slices
print(f"Slices — original: {original_slices}, augmented: {augmented_slices}, total: {total_slices}")

# Split 80% train, 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# Model
model = BrainCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Track metrics
history = {"loss": [], "val_accuracy": [], "val_labels": [], "val_preds": [],
           "original_slices": original_slices, "augmented_slices": augmented_slices}

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    subject_preds = defaultdict(list)
    subject_labels = {}

    with torch.no_grad():
        for images, labels, subject_ids in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for sid, pred, lbl in zip(subject_ids, predicted.tolist(), labels.tolist()):
                subject_preds[sid].append(pred)
                subject_labels[sid] = lbl

    all_labels = list(subject_labels.values())
    all_preds = [Counter(subject_preds[sid]).most_common(1)[0][0] for sid in subject_labels]
    acc = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    history["loss"].append(total_loss)
    history["val_accuracy"].append(acc)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Subject Accuracy: {acc:.2f}%")

# Save final epoch predictions for confusion matrix
history["val_labels"] = all_labels
history["val_preds"] = all_preds

# Save model and metrics
torch.save(model.state_dict(), "models/brain_cnn.pth")
with open("models/history.json", "w") as f:
    json.dump(history, f)

print("\nModel saved → models/brain_cnn.pth")
print("Metrics saved → models/history.json")
print("\nPer-class report (subject-level):")
print(classification_report(all_labels, all_preds, target_names=["HC", "AVH-", "AVH+"]))