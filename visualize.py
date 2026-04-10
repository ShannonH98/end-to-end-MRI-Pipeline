import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import torch
from models.cnn import BrainCNN
from preprocessing.dataset import BrainSliceDataset
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

# Config
SLICES = "data/slices"
TSV = "data/participants.tsv"
HISTORY = "models/history.json"
MODEL = "models/brain_cnn.pth"
CLASS_NAMES = ["HC", "AVH+"]

# Load metrics
with open(HISTORY) as f:
    history = json.load(f)

# --- 1. Loss Curve ---
def plot_loss(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history["loss"], marker="o", color="steelblue")
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("models/loss_curve.png")
    plt.show()
    print("Saved: models/loss_curve.png")

# --- 2. Accuracy Curve ---
def plot_accuracy(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history["val_accuracy"], marker="o", color="seagreen")
    plt.title("Validation Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("models/accuracy_curve.png")
    plt.show()
    print("Saved: models/accuracy_curve.png")

# --- 3. Confusion Matrix ---
def plot_confusion_matrix(history):
    cm = confusion_matrix(history["val_labels"], history["val_preds"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png")
    plt.show()
    print("Saved: models/confusion_matrix.png")

# --- 4. Sample Brain Slices ---
def plot_sample_slices():
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    classes = ["sub-01_T1w", "sub-73_T1w"]
    labels = ["HC", "AVH+"]

    for row, (subject, label) in enumerate(zip(classes, labels)):
        folder = os.path.join(SLICES, subject)
        files = sorted(os.listdir(folder))[::10][:5]
        for col, fname in enumerate(files):
            img = Image.open(os.path.join(folder, fname))
            axes[row][col].imshow(img, cmap="gray")
            axes[row][col].axis("off")
            axes[row][col].set_title(f"{label}\n{fname}", fontsize=7)

    plt.suptitle("Sample Brain Slices — HC vs AVH+")
    plt.tight_layout()
    plt.savefig("models/sample_slices.png")
    plt.show()
    print("Saved: models/sample_slices.png")

# --- 5. Grad-CAM Heatmap ---
def plot_gradcam():
    model = BrainCNN()
    model.load_state_dict(torch.load(MODEL))
    model.eval()

    dataset = BrainSliceDataset(SLICES, TSV)
    img, label = dataset[len(dataset) // 2]
    input_tensor = img.unsqueeze(0).requires_grad_(True)

    # Forward pass
    features = None
    grads = None

    def forward_hook(module, input, output):
        nonlocal features
        features = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal grads
        grads = grad_output[0]

    # Hook into last conv layer
    handle_f = model.conv[-1].register_forward_hook(forward_hook)
    handle_b = model.conv[-1].register_full_backward_hook(backward_hook)

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    output[0, pred_class].backward()

    handle_f.remove()
    handle_b.remove()

    # Compute Grad-CAM
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * features).sum(dim=1).squeeze()
    cam = F.relu(cam)
    cam = cam.detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    # Original image
    orig = img.squeeze().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(orig, cmap="gray")
    axes[0].set_title(f"Original — True: {CLASS_NAMES[label]}")
    axes[0].axis("off")

    axes[1].imshow(orig, cmap="gray")
    axes[1].imshow(cam, cmap="jet", alpha=0.4, extent=[0, orig.shape[1], orig.shape[0], 0])
    axes[1].set_title(f"Grad-CAM — Predicted: {CLASS_NAMES[pred_class]}")
    axes[1].axis("off")

    plt.suptitle("Grad-CAM Heatmap")
    plt.tight_layout()
    plt.savefig("models/gradcam.png")
    plt.show()
    print("Saved: models/gradcam.png")

# --- Run all ---
if __name__ == "__main__":
    print("Generating visualisations...")
    plot_loss(history)
    plot_accuracy(history)
    plot_confusion_matrix(history)
    plot_sample_slices()
    plot_gradcam()
    print("\nAll visualisations saved to models/")