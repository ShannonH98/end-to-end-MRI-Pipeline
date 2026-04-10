import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from preprocessing.dataset import BrainSliceDataset
from models.cnn import BrainCNN
import json

# Config
SLICES = "data/slices"
TSV = "data/participants.tsv"
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Dataset
dataset = BrainSliceDataset(SLICES, TSV)

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
history = {"loss": [], "val_accuracy": [], "val_labels": [], "val_preds": []}

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
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

    acc = 100 * correct / total
    history["loss"].append(total_loss)
    history["val_accuracy"].append(acc)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Val Accuracy: {acc:.2f}%")

# Save final epoch predictions for confusion matrix
history["val_labels"] = all_labels
history["val_preds"] = all_preds

# Save model and metrics
torch.save(model.state_dict(), "models/brain_cnn.pth")
with open("models/history.json", "w") as f:
    json.dump(history, f)

print("\nModel saved → models/brain_cnn.pth")
print("Metrics saved → models/history.json")