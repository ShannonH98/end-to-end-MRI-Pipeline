import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

RESULTS = "models/comparison_results.json"
CLASS_NAMES = ["HC", "AVH-", "AVH+"]

with open(RESULTS) as f:
    r = json.load(f)

cnn_hist     = r["cnn"]["history"]
resnet_hist  = r["resnet"]["history"]
cnn_test_labels   = r["cnn"]["test_labels"]
cnn_test_preds    = r["cnn"]["test_preds"]
resnet_test_labels = r["resnet"]["test_labels"]
resnet_test_preds  = r["resnet"]["test_preds"]

cnn_test_acc    = 100 * sum(p == l for p, l in zip(cnn_test_preds, cnn_test_labels)) / len(cnn_test_labels)
resnet_test_acc = 100 * sum(p == l for p, l in zip(resnet_test_preds, resnet_test_labels)) / len(resnet_test_labels)

epochs = list(range(1, len(cnn_hist["loss"]) + 1))

# --- 1. Loss curves ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epochs, cnn_hist["loss"],    marker="o", label="BrainCNN",    color="steelblue")
ax.plot(epochs, resnet_hist["loss"], marker="s", label="BrainResNet", color="darkorange")
ax.set_title("Training Loss per Epoch")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("models/comparison_loss.png", dpi=150)
plt.show()
print("Saved: models/comparison_loss.png")

# --- 2. Val accuracy curves ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epochs, cnn_hist["val_accuracy"],    marker="o", label="BrainCNN",    color="steelblue")
ax.plot(epochs, resnet_hist["val_accuracy"], marker="s", label="BrainResNet", color="darkorange")
ax.axhline(y=33.33, color="gray", linestyle="--", linewidth=1, label="Chance (33%)")
ax.set_title("Validation Accuracy per Epoch (Subject-Level)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("models/comparison_val_accuracy.png", dpi=150)
plt.show()
print("Saved: models/comparison_val_accuracy.png")

# --- 3. Test confusion matrices ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, labels, preds, title in [
    (axes[0], cnn_test_labels,    cnn_test_preds,    f"BrainCNN  ({cnn_test_acc:.1f}% test)"),
    (axes[1], resnet_test_labels, resnet_test_preds, f"BrainResNet  ({resnet_test_acc:.1f}% test)"),
]:
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)

plt.suptitle("Test Set Confusion Matrices (Subject-Level)", fontsize=13)
plt.tight_layout()
plt.savefig("models/comparison_confusion_matrices.png", dpi=150)
plt.show()
print("Saved: models/comparison_confusion_matrices.png")

# --- 4. Summary table ---
print("\n" + "=" * 55)
print("MODEL COMPARISON (subject-level, no data leakage)")
print("=" * 55)
print(f"\n{'Metric':<22} {'BrainCNN':>14} {'BrainResNet':>14}")
print("-" * 55)
print(f"{'Best Val Acc':<22} {max(cnn_hist['val_accuracy']):>13.2f}% {max(resnet_hist['val_accuracy']):>13.2f}%")
print(f"{'Test Acc':<22} {cnn_test_acc:>13.2f}% {resnet_test_acc:>13.2f}%")
print(f"{'Chance baseline':<22} {'33.33%':>14} {'33.33%':>14}")

print("\n--- BrainCNN test report ---")
print(classification_report(cnn_test_labels, cnn_test_preds, target_names=CLASS_NAMES, zero_division=0))

print("--- BrainResNet test report ---")
print(classification_report(resnet_test_labels, resnet_test_preds, target_names=CLASS_NAMES, zero_division=0))
