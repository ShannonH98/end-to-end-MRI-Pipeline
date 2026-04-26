import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict, Counter
from sklearn.metrics import classification_report
from preprocessing.label_map import load_labels
from models.cnn import BrainCNN
from models.BrainResNet import BrainResNet

# Config
SLICES = "data/slices"
TSV = "data/participants.tsv"
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SEED = 42
CLASS_NAMES = ["HC", "AVH-", "AVH+"]

class BrainDataset(Dataset):
    def __init__(self, slices_folder, tsv_path, image_size, grayscale_channels=1):
        self.samples = []
        channel_transform = transforms.Grayscale(num_output_channels=grayscale_channels)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ) if grayscale_channels == 3 else transforms.Normalize(mean=[0.5], std=[0.5])

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            channel_transform,
            transforms.ToTensor(),
            normalize,
        ])

        labels = load_labels(tsv_path)

        for subject in os.listdir(slices_folder):
            subject_path = os.path.join(slices_folder, subject)
            if not os.path.isdir(subject_path):
                continue
            pid = "sub-" + subject.split("-")[1].split("_")[0]
            if pid not in labels:
                continue
            for slice_file in sorted(os.listdir(subject_path)):
                if slice_file.endswith(".png"):
                    self.samples.append((os.path.join(subject_path, slice_file), labels[pid], pid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, subject_id = self.samples[idx]
        img = Image.open(path).convert("L")
        img = self.transform(img)
        return img, label, subject_id


def train_and_evaluate(model, train_loader, val_loader, optimizer, label):
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
        print(f"  [{label}] Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Subject Acc: {acc:.2f}%")

    return all_labels, all_preds, history


# --- Build datasets with fixed seed for identical splits ---
torch.manual_seed(SEED)
cnn_dataset = BrainDataset(SLICES, TSV, image_size=128, grayscale_channels=1)
train_size = int(0.8 * len(cnn_dataset))
val_size = len(cnn_dataset) - train_size
cnn_train, cnn_val = random_split(cnn_dataset, [train_size, val_size])

torch.manual_seed(SEED)
resnet_dataset = BrainDataset(SLICES, TSV, image_size=224, grayscale_channels=3)
resnet_train, resnet_val = random_split(resnet_dataset, [train_size, val_size])

cnn_train_loader = DataLoader(cnn_train, batch_size=BATCH_SIZE, shuffle=True)
cnn_val_loader = DataLoader(cnn_val, batch_size=BATCH_SIZE)
resnet_train_loader = DataLoader(resnet_train, batch_size=BATCH_SIZE, shuffle=True)
resnet_val_loader = DataLoader(resnet_val, batch_size=BATCH_SIZE)

# --- Train BrainCNN ---
print("\n=== Training BrainCNN ===")
cnn_model = BrainCNN()
cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
cnn_labels, cnn_preds, cnn_history = train_and_evaluate(cnn_model, cnn_train_loader, cnn_val_loader, cnn_optimizer, "CNN")

# --- Train BrainResNet ---
print("\n=== Training BrainResNet ===")
resnet_model = BrainResNet()
resnet_optimizer = torch.optim.Adam(resnet_model.model.fc.parameters(), lr=LEARNING_RATE)
resnet_labels, resnet_preds, resnet_history = train_and_evaluate(resnet_model, resnet_train_loader, resnet_val_loader, resnet_optimizer, "ResNet")

# --- Side-by-side comparison ---
print("\n" + "="*55)
print("MODEL COMPARISON (subject-level, same data split)")
print("="*55)

print(f"\n{'Metric':<20} {'BrainCNN':>15} {'BrainResNet':>15}")
print("-"*55)
print(f"{'Final Accuracy':<20} {cnn_history['val_accuracy'][-1]:>14.2f}% {resnet_history['val_accuracy'][-1]:>14.2f}%")
print(f"{'Best Accuracy':<20} {max(cnn_history['val_accuracy']):>14.2f}% {max(resnet_history['val_accuracy']):>14.2f}%")

print("\n--- BrainCNN per-class report ---")
print(classification_report(cnn_labels, cnn_preds, target_names=CLASS_NAMES))

print("--- BrainResNet per-class report ---")
print(classification_report(resnet_labels, resnet_preds, target_names=CLASS_NAMES))

# --- Save results ---
results = {
    "seed": SEED,
    "epochs": EPOCHS,
    "cnn": {"history": cnn_history, "val_labels": cnn_labels, "val_preds": cnn_preds},
    "resnet": {"history": resnet_history, "val_labels": resnet_labels, "val_preds": resnet_preds}
}
with open("models/comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)

torch.save(cnn_model.state_dict(), "models/brain_cnn.pth")
torch.save(resnet_model.state_dict(), "models/brain_resnet.pth")
print("Results saved → models/comparison_results.json")
