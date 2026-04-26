import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict, Counter
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.label_map import load_labels

# Config
SLICES = "data/slices"
TSV = "data/participants.tsv"
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

class BrainResNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = resnet18(weights="IMAGENET1K_V1")
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)

class BrainSliceDataset(Dataset):
    def __init__(self, slices_folder, tsv_path):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        labels = load_labels(tsv_path)

        for subject in os.listdir(slices_folder):
            subject_path = os.path.join(slices_folder, subject)
            if not os.path.isdir(subject_path):
                continue

            pid = "sub-" + subject.split("-")[1].split("_")[0]

            if pid not in labels:
                print(f"Skipping {subject} — not in participants.tsv")
                continue

            label = labels[pid]

            for slice_file in sorted(os.listdir(subject_path)):
                if slice_file.endswith(".png"):
                    self.samples.append((
                        os.path.join(subject_path, slice_file),
                        label,
                        pid
                    ))

        print(f"Dataset ready: {len(self.samples)} slices")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, subject_id = self.samples[idx]
        img = Image.open(path).convert("L")
        img = self.transform(img)
        return img, label, subject_id

if __name__ == "__main__":
    dataset = BrainSliceDataset(SLICES, TSV)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = BrainResNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.model.fc.parameters(), lr=LEARNING_RATE)

    history = {"loss": [], "val_accuracy": [], "val_labels": [], "val_preds": []}

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

    history["val_labels"] = all_labels
    history["val_preds"] = all_preds

    torch.save(model.state_dict(), "models/brain_resnet.pth")
    with open("models/resnet_history.json", "w") as f:
        json.dump(history, f)

    print("\nModel saved → models/brain_resnet.pth")
    print("Metrics saved → models/resnet_history.json")
