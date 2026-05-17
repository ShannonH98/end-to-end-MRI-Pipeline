import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

RESULTS = "models/kfold_results.json"
CLASS_NAMES = ["HC", "AVH-", "AVH+"]

with open(RESULTS) as f:
    r = json.load(f)

n_folds = r["n_folds"]
folds = list(range(1, n_folds + 1))

cnn_fold_accs    = r["cnn"]["fold_accs"]
resnet_fold_accs = r["resnet"]["fold_accs"]
cnn_mean,    cnn_std    = r["cnn"]["mean"],    r["cnn"]["std"]
resnet_mean, resnet_std = r["resnet"]["mean"], r["resnet"]["std"]

# Aggregate labels/preds across all folds for confusion matrices
cnn_all_labels    = [l for fold in r["cnn"]["folds"]    for l in fold["test_labels"]]
cnn_all_preds     = [p for fold in r["cnn"]["folds"]    for p in fold["test_preds"]]
resnet_all_labels = [l for fold in r["resnet"]["folds"] for l in fold["test_labels"]]
resnet_all_preds  = [p for fold in r["resnet"]["folds"] for p in fold["test_preds"]]

# --- 1. Per-fold accuracy bar chart ---
x = np.arange(n_folds)
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, cnn_fold_accs,    width, label="BrainCNN",    color="steelblue",  alpha=0.85)
bars2 = ax.bar(x + width/2, resnet_fold_accs, width, label="BrainResNet", color="darkorange", alpha=0.85)
ax.axhline(y=33.33,       color="gray",     linestyle="--", linewidth=1, label="Chance (33%)")
ax.axhline(y=cnn_mean,    color="steelblue",  linestyle=":",  linewidth=1.5, label=f"CNN mean ({cnn_mean:.1f}%)")
ax.axhline(y=resnet_mean, color="darkorange", linestyle=":",  linewidth=1.5, label=f"ResNet mean ({resnet_mean:.1f}%)")
ax.set_title("Per-Fold Test Accuracy (Subject-Level, 5-Fold CV)")
ax.set_xlabel("Fold")
ax.set_ylabel("Accuracy (%)")
ax.set_xticks(x)
ax.set_xticklabels([f"Fold {i}" for i in folds])
ax.set_ylim(0, 100)
ax.legend()
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("models/kfold_accuracy.png", dpi=150)
plt.show()
print("Saved: models/kfold_accuracy.png")

# --- 2. Aggregated confusion matrices ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, labels, preds, name, mean in [
    (axes[0], cnn_all_labels,    cnn_all_preds,    "BrainCNN",    cnn_mean),
    (axes[1], resnet_all_labels, resnet_all_preds, "BrainResNet", resnet_mean),
]:
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}  (mean {mean:.1f}%)")

plt.suptitle("Aggregated Confusion Matrices — All Folds (Subject-Level)", fontsize=13)
plt.tight_layout()
plt.savefig("models/kfold_confusion_matrices.png", dpi=150)
plt.show()
print("Saved: models/kfold_confusion_matrices.png")

# --- 3. Summary table ---
print("\n" + "=" * 58)
print("K-FOLD SUMMARY (5-fold, subject-level majority vote)")
print("=" * 58)
print(f"\n{'Fold':<22} {'BrainCNN':>14} {'BrainResNet':>14}")
print("-" * 58)
for i, (ca, ra) in enumerate(zip(cnn_fold_accs, resnet_fold_accs)):
    print(f"  Fold {i+1:<17} {ca:>13.2f}% {ra:>13.2f}%")
print("-" * 58)
print(f"{'Mean Accuracy':<22} {cnn_mean:>13.2f}% {resnet_mean:>13.2f}%")
print(f"{'Std Dev':<22} {cnn_std:>13.2f}% {resnet_std:>13.2f}%")
print(f"{'Chance baseline':<22} {'33.33%':>14} {'33.33%':>14}")

print("\n--- BrainCNN aggregated report ---")
print(classification_report(cnn_all_labels, cnn_all_preds, target_names=CLASS_NAMES, zero_division=0))

print("--- BrainResNet aggregated report ---")
print(classification_report(resnet_all_labels, resnet_all_preds, target_names=CLASS_NAMES, zero_division=0))
