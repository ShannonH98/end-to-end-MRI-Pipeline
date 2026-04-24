import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.BrainResNet import BrainResNet, BrainSliceDataset

# Config
SLICES = "data/slices"
TSV = "data/participants.tsv"
HISTORY = "models/resnet_history.json"
MODEL = "models/brain_resnet.pth"
CLASS_NAMES = ["HC", "AVH-", "AVH+"]

with open(HISTORY) as f:
    history = json.load(f)

# --- 1. Loss Curve ---
def plot_loss(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history["loss"], marker="o", color="steelblue")
    plt.title("ResNet Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("models/resnet_loss_curve.png")
    plt.show()
    print("Saved: models/resnet_loss_curve.png")

# --- 2. Accuracy Curve ---
def plot_accuracy(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history["val_accuracy"], marker="o", color="seagreen")
    plt.title("ResNet Validation Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("models/resnet_accuracy_curve.png")
    plt.show()
    print("Saved: models/resnet_accuracy_curve.png")

# --- 3. Confusion Matrix ---
def plot_confusion_matrix(history):
    cm = confusion_matrix(history["val_labels"], history["val_preds"], labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("ResNet Confusion Matrix")
    plt.tight_layout()
    plt.savefig("models/resnet_confusion_matrix.png")
    plt.show()
    print("Saved: models/resnet_confusion_matrix.png")

# --- 4. Sample Brain Slices ---
def plot_sample_slices():
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    classes = ["sub-01_T1w", "sub-26_T1w", "sub-73_T1w"]
    labels = ["HC", "AVH-", "AVH+"]

    for row, (subject, label) in enumerate(zip(classes, labels)):
        folder = os.path.join(SLICES, subject)
        files = sorted(os.listdir(folder))[::10][:5]
        for col, fname in enumerate(files):
            img = Image.open(os.path.join(folder, fname))
            axes[row][col].imshow(img, cmap="gray")
            axes[row][col].axis("off")
            axes[row][col].set_title(f"{label}\n{fname}", fontsize=7)

    plt.suptitle("Sample Brain Slices — HC vs AVH- vs AVH+")
    plt.tight_layout()
    plt.savefig("models/resnet_sample_slices.png")
    plt.show()
    print("Saved: models/resnet_sample_slices.png")

# --- 5. Grad-CAM Heatmap ---
def plot_gradcam():
    model = BrainResNet()
    model.load_state_dict(torch.load(MODEL))
    model.eval()

    dataset = BrainSliceDataset(SLICES, TSV)
    img, label = dataset[len(dataset) // 2]
    input_tensor = img.unsqueeze(0).requires_grad_(True)

    features = None
    grads = None

    def forward_hook(module, input, output):
        nonlocal features
        features = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal grads
        grads = grad_output[0]

    # Hook into ResNet's last conv block
    handle_f = model.model.layer4[-1].register_forward_hook(forward_hook)
    handle_b = model.model.layer4[-1].register_full_backward_hook(backward_hook)

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    output[0, pred_class].backward()

    handle_f.remove()
    handle_b.remove()

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * features).sum(dim=1).squeeze()
    cam = F.relu(cam)
    cam = cam.detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    # Show first channel of 3-channel input as grayscale
    orig = img[0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(orig, cmap="gray")
    axes[0].set_title(f"Original — True: {CLASS_NAMES[label]}")
    axes[0].axis("off")

    axes[1].imshow(orig, cmap="gray")
    axes[1].imshow(cam, cmap="jet", alpha=0.4, extent=[0, orig.shape[1], orig.shape[0], 0])
    axes[1].set_title(f"Grad-CAM — Predicted: {CLASS_NAMES[pred_class]}")
    axes[1].axis("off")

    plt.suptitle("ResNet Grad-CAM Heatmap")
    plt.tight_layout()
    plt.savefig("models/resnet_gradcam.png")
    plt.show()
    print("Saved: models/resnet_gradcam.png")

# --- Run all ---
if __name__ == "__main__":
    print("Generating ResNet visualisations...")
    plot_loss(history)
    plot_accuracy(history)
    plot_confusion_matrix(history)
    plot_sample_slices()
    plot_gradcam()
    print("\nAll visualisations saved to models/")
