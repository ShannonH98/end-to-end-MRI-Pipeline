import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

RESULTS = "models/kfold_results.json"
CLASS_NAMES = ["HC", "AVH-", "AVH+"]

with open(RESULTS) as f:
    r = json.load(f)

n_folds  = r["n_folds"]
epoch_x  = list(range(1, r["epochs"] + 1))
folds    = list(range(1, n_folds + 1))

cnn_fold_accs    = r["cnn"]["fold_accs"]
resnet_fold_accs = r["resnet"]["fold_accs"]
cnn_mean,    cnn_std    = r["cnn"]["mean"],    r["cnn"]["std"]
resnet_mean, resnet_std = r["resnet"]["mean"], r["resnet"]["std"]

use_rad = "radimagenet" in r
if use_rad:
    rad_fold_accs        = r["radimagenet"]["fold_accs"]
    rad_mean, rad_std    = r["radimagenet"]["mean"], r["radimagenet"]["std"]
    rad_all_labels       = [l for fold in r["radimagenet"]["folds"] for l in fold["test_labels"]]
    rad_all_preds        = [p for fold in r["radimagenet"]["folds"] for p in fold["test_preds"]]

# Aggregate labels/preds across all folds
cnn_all_labels    = [l for fold in r["cnn"]["folds"]    for l in fold["test_labels"]]
cnn_all_preds     = [p for fold in r["cnn"]["folds"]    for p in fold["test_preds"]]
resnet_all_labels = [l for fold in r["resnet"]["folds"] for l in fold["test_labels"]]
resnet_all_preds  = [p for fold in r["resnet"]["folds"] for p in fold["test_preds"]]

# --- 1. Loss curves (mean across folds) ---
cnn_loss    = np.mean([[f["history"]["loss"][e] for f in r["cnn"]["folds"]]    for e in range(r["epochs"])], axis=1)
resnet_loss = np.mean([[f["history"]["loss"][e] for f in r["resnet"]["folds"]] for e in range(r["epochs"])], axis=1)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epoch_x, cnn_loss,    marker="o", label="BrainCNN",    color="steelblue")
ax.plot(epoch_x, resnet_loss, marker="s", label="BrainResNet", color="darkorange")
if use_rad:
    rad_loss = np.mean([[f["history"]["loss"][e] for f in r["radimagenet"]["folds"]] for e in range(r["epochs"])], axis=1)
    ax.plot(epoch_x, rad_loss, marker="^", label="RadImageNet", color="seagreen")
ax.set_title("Mean Training Loss per Epoch (Averaged Across 5 Folds)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("models/kfold_loss.png", dpi=150)
plt.show()
print("Saved: models/kfold_loss.png")

# --- 2. Accuracy curves (mean across folds) ---
cnn_acc    = np.mean([[f["history"]["accuracy"][e] for f in r["cnn"]["folds"]]    for e in range(r["epochs"])], axis=1)
resnet_acc = np.mean([[f["history"]["accuracy"][e] for f in r["resnet"]["folds"]] for e in range(r["epochs"])], axis=1)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epoch_x, cnn_acc,    marker="o", label="BrainCNN",    color="steelblue")
ax.plot(epoch_x, resnet_acc, marker="s", label="BrainResNet", color="darkorange")
if use_rad:
    rad_acc = np.mean([[f["history"]["accuracy"][e] for f in r["radimagenet"]["folds"]] for e in range(r["epochs"])], axis=1)
    ax.plot(epoch_x, rad_acc, marker="^", label="RadImageNet", color="seagreen")
ax.axhline(y=33.33, color="gray", linestyle="--", linewidth=1, label="Chance (33%)")
ax.set_title("Mean Test Accuracy per Epoch (Averaged Across 5 Folds)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("models/kfold_accuracy_curve.png", dpi=150)
plt.show()
print("Saved: models/kfold_accuracy_curve.png")

# --- 3. Per-fold accuracy bar chart ---
x     = np.arange(n_folds)
width = 0.25 if use_rad else 0.35

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width if use_rad else x - width/2, cnn_fold_accs,    width, label="BrainCNN",    color="steelblue",  alpha=0.85)
ax.bar(x,                                      resnet_fold_accs, width, label="BrainResNet", color="darkorange", alpha=0.85)
if use_rad:
    ax.bar(x + width, rad_fold_accs, width, label="RadImageNet", color="seagreen", alpha=0.85)
ax.axhline(y=33.33,       color="gray",       linestyle="--", linewidth=1,   label="Chance (33%)")
ax.axhline(y=cnn_mean,    color="steelblue",  linestyle=":",  linewidth=1.5, label=f"CNN mean ({cnn_mean:.1f}%)")
ax.axhline(y=resnet_mean, color="darkorange", linestyle=":",  linewidth=1.5, label=f"ResNet mean ({resnet_mean:.1f}%)")
if use_rad:
    ax.axhline(y=rad_mean, color="seagreen", linestyle=":", linewidth=1.5, label=f"RadImageNet mean ({rad_mean:.1f}%)")
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

# --- 4. Aggregated confusion matrices ---
n_models = 3 if use_rad else 2
fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

models_to_plot = [
    (axes[0], cnn_all_labels,    cnn_all_preds,    "BrainCNN",    cnn_mean),
    (axes[1], resnet_all_labels, resnet_all_preds, "BrainResNet", resnet_mean),
]
if use_rad:
    models_to_plot.append((axes[2], rad_all_labels, rad_all_preds, "RadImageNet", rad_mean))

for ax, labels, preds, name, mean in models_to_plot:
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}  (mean {mean:.1f}%)")

plt.suptitle("Aggregated Confusion Matrices — All Folds (Subject-Level)", fontsize=13)
plt.tight_layout()
plt.savefig("models/kfold_confusion_matrices.png", dpi=150)
plt.show()
print("Saved: models/kfold_confusion_matrices.png")

# --- 5. Summary table ---
print("\n" + "=" * 65)
print("K-FOLD SUMMARY (5-fold, subject-level majority vote)")
print("=" * 65)
header = f"\n{'Fold':<22} {'BrainCNN':>14} {'BrainResNet':>14}"
if use_rad:
    header += f" {'RadImageNet':>14}"
print(header)
print("-" * 65)
for i, (ca, ra) in enumerate(zip(cnn_fold_accs, resnet_fold_accs)):
    row = f"  Fold {i+1:<17} {ca:>13.2f}% {ra:>13.2f}%"
    if use_rad:
        row += f" {rad_fold_accs[i]:>13.2f}%"
    print(row)
print("-" * 65)
mean_row = f"{'Mean Accuracy':<22} {cnn_mean:>13.2f}% {resnet_mean:>13.2f}%"
std_row  = f"{'Std Dev':<22} {cnn_std:>13.2f}% {resnet_std:>13.2f}%"
if use_rad:
    mean_row += f" {rad_mean:>13.2f}%"
    std_row  += f" {rad_std:>13.2f}%"
print(mean_row)
print(std_row)
print(f"{'Chance baseline':<22} {'33.33%':>14} {'33.33%':>14}")

print("\n--- BrainCNN aggregated report ---")
print(classification_report(cnn_all_labels, cnn_all_preds, target_names=CLASS_NAMES, zero_division=0))
print("--- BrainResNet aggregated report ---")
print(classification_report(resnet_all_labels, resnet_all_preds, target_names=CLASS_NAMES, zero_division=0))
if use_rad:
    print("--- RadImageNet aggregated report ---")
    print(classification_report(rad_all_labels, rad_all_preds, target_names=CLASS_NAMES, zero_division=0))
