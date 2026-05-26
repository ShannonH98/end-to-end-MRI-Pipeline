import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

OASIS_RESULTS  = "models/oasis_results.json"
ORIG_RESULTS   = "models/kfold_results.json"
OASIS_CLASSES  = ["Non Demented", "Very Mild Dementia", "Mild Dementia", "Moderate Dementia"]
ORIG_CLASSES   = ["HC", "AVH-", "AVH+"]
OASIS_CHANCE   = 25.0
ORIG_CHANCE    = 33.33

with open(OASIS_RESULTS) as f:
    o = json.load(f)

n_folds  = o["n_folds"]
epochs   = o["epochs"]
epoch_x  = list(range(1, epochs + 1))
folds    = list(range(1, n_folds + 1))

cnn_fold_accs    = o["oasis_cnn"]["fold_accs"]
resnet_fold_accs = o["resnet"]["fold_accs"]
cnn_mean,    cnn_std    = o["oasis_cnn"]["mean"], o["oasis_cnn"]["std"]
resnet_mean, resnet_std = o["resnet"]["mean"],    o["resnet"]["std"]

use_rad = "radimagenet" in o
if use_rad:
    rad_fold_accs     = o["radimagenet"]["fold_accs"]
    rad_mean, rad_std = o["radimagenet"]["mean"], o["radimagenet"]["std"]
    rad_all_labels    = [l for fold in o["radimagenet"]["folds"] for l in fold["test_labels"]]
    rad_all_preds     = [p for fold in o["radimagenet"]["folds"] for p in fold["test_preds"]]

cnn_all_labels    = [l for fold in o["oasis_cnn"]["folds"] for l in fold["test_labels"]]
cnn_all_preds     = [p for fold in o["oasis_cnn"]["folds"] for p in fold["test_preds"]]
resnet_all_labels = [l for fold in o["resnet"]["folds"]    for l in fold["test_labels"]]
resnet_all_preds  = [p for fold in o["resnet"]["folds"]    for p in fold["test_preds"]]


# --- 1. Loss curves ---
cnn_loss    = np.mean([[f["history"]["loss"][e] for f in o["oasis_cnn"]["folds"]] for e in range(epochs)], axis=1)
resnet_loss = np.mean([[f["history"]["loss"][e] for f in o["resnet"]["folds"]]    for e in range(epochs)], axis=1)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epoch_x, cnn_loss,    marker="o", label="OasisCNN",    color="steelblue")
ax.plot(epoch_x, resnet_loss, marker="s", label="BrainResNet", color="darkorange")
if use_rad:
    rad_loss = np.mean([[f["history"]["loss"][e] for f in o["radimagenet"]["folds"]] for e in range(epochs)], axis=1)
    ax.plot(epoch_x, rad_loss, marker="^", label="RadImageNet", color="seagreen")
ax.set_title("OASIS — Mean Training Loss per Epoch (Averaged Across 5 Folds)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("models/oasis_loss.png", dpi=150)
plt.show()
print("Saved: models/oasis_loss.png")


# --- 2. Accuracy curves ---
cnn_acc_curve    = np.mean([[f["history"]["accuracy"][e] for f in o["oasis_cnn"]["folds"]] for e in range(epochs)], axis=1)
resnet_acc_curve = np.mean([[f["history"]["accuracy"][e] for f in o["resnet"]["folds"]]    for e in range(epochs)], axis=1)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epoch_x, cnn_acc_curve,    marker="o", label="OasisCNN",    color="steelblue")
ax.plot(epoch_x, resnet_acc_curve, marker="s", label="BrainResNet", color="darkorange")
if use_rad:
    rad_acc_curve = np.mean([[f["history"]["accuracy"][e] for f in o["radimagenet"]["folds"]] for e in range(epochs)], axis=1)
    ax.plot(epoch_x, rad_acc_curve, marker="^", label="RadImageNet", color="seagreen")
ax.axhline(y=OASIS_CHANCE, color="gray", linestyle="--", linewidth=1, label=f"Chance ({OASIS_CHANCE}%)")
ax.set_title("OASIS — Mean Test Accuracy per Epoch (Averaged Across 5 Folds)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("models/oasis_accuracy_curve.png", dpi=150)
plt.show()
print("Saved: models/oasis_accuracy_curve.png")


# --- 3. Per-fold accuracy bar chart ---
x     = np.arange(n_folds)
width = 0.25 if use_rad else 0.35

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width if use_rad else x - width/2, cnn_fold_accs,    width, label="OasisCNN",    color="steelblue",  alpha=0.85)
ax.bar(x,                                      resnet_fold_accs, width, label="BrainResNet", color="darkorange", alpha=0.85)
if use_rad:
    ax.bar(x + width, rad_fold_accs, width, label="RadImageNet", color="seagreen", alpha=0.85)
ax.axhline(y=OASIS_CHANCE, color="gray",       linestyle="--", linewidth=1,   label=f"Chance ({OASIS_CHANCE}%)")
ax.axhline(y=cnn_mean,     color="steelblue",  linestyle=":",  linewidth=1.5, label=f"OasisCNN mean ({cnn_mean:.1f}%)")
ax.axhline(y=resnet_mean,  color="darkorange", linestyle=":",  linewidth=1.5, label=f"ResNet mean ({resnet_mean:.1f}%)")
if use_rad:
    ax.axhline(y=rad_mean, color="seagreen", linestyle=":", linewidth=1.5, label=f"RadImageNet mean ({rad_mean:.1f}%)")
ax.set_title("OASIS — Per-Fold Test Accuracy (Subject-Level, 5-Fold CV)")
ax.set_xlabel("Fold")
ax.set_ylabel("Accuracy (%)")
ax.set_xticks(x)
ax.set_xticklabels([f"Fold {i}" for i in folds])
ax.set_ylim(0, 100)
ax.legend()
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("models/oasis_accuracy.png", dpi=150)
plt.show()
print("Saved: models/oasis_accuracy.png")


# --- 4. Confusion matrices ---
n_models = 3 if use_rad else 2
fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

models_to_plot = [
    (axes[0], cnn_all_labels,    cnn_all_preds,    "OasisCNN",    cnn_mean),
    (axes[1], resnet_all_labels, resnet_all_preds, "BrainResNet", resnet_mean),
]
if use_rad:
    models_to_plot.append((axes[2], rad_all_labels, rad_all_preds, "RadImageNet", rad_mean))

short_names = ["Non Dem.", "V.Mild", "Mild", "Moderate"]
for ax, labels, preds, name, mean in models_to_plot:
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=short_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}  (mean {mean:.1f}%)")

plt.suptitle("OASIS — Aggregated Confusion Matrices — All Folds (Subject-Level)", fontsize=13)
plt.tight_layout()
plt.savefig("models/oasis_confusion_matrices.png", dpi=150)
plt.show()
print("Saved: models/oasis_confusion_matrices.png")


# --- 5. Cross-dataset comparison (ResNet & RadImageNet, normalized above chance) ---
import os
if os.path.exists(ORIG_RESULTS):
    with open(ORIG_RESULTS) as f:
        orig = json.load(f)

    orig_resnet_mean = orig["resnet"]["mean"]
    orig_resnet_std  = orig["resnet"]["std"]
    oasis_resnet_mean = resnet_mean
    oasis_resnet_std  = resnet_std

    models_compare = ["BrainResNet"]
    orig_means  = [orig_resnet_mean]
    oasis_means = [oasis_resnet_mean]
    orig_stds   = [orig_resnet_std]
    oasis_stds  = [oasis_resnet_std]
    orig_above  = [orig_resnet_mean  - ORIG_CHANCE]
    oasis_above = [oasis_resnet_mean - OASIS_CHANCE]

    if use_rad and "radimagenet" in orig:
        models_compare.append("RadImageNet")
        orig_means.append(orig["radimagenet"]["mean"])
        oasis_means.append(o["radimagenet"]["mean"])
        orig_stds.append(orig["radimagenet"]["std"])
        oasis_stds.append(o["radimagenet"]["std"])
        orig_above.append(orig["radimagenet"]["mean"]  - ORIG_CHANCE)
        oasis_above.append(o["radimagenet"]["mean"] - OASIS_CHANCE)

    x      = np.arange(len(models_compare))
    width  = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Raw accuracy
    ax = axes[0]
    ax.bar(x - width/2, orig_means,  width, yerr=orig_stds,  label="Original dataset (3-class)", color="royalblue",   alpha=0.85, capsize=5)
    ax.bar(x + width/2, oasis_means, width, yerr=oasis_stds, label="OASIS dataset (4-class)",    color="darkorange",  alpha=0.85, capsize=5)
    ax.axhline(y=ORIG_CHANCE,  color="royalblue",  linestyle="--", linewidth=1, label=f"Orig. chance ({ORIG_CHANCE}%)")
    ax.axhline(y=OASIS_CHANCE, color="darkorange", linestyle="--", linewidth=1, label=f"OASIS chance ({OASIS_CHANCE}%)")
    ax.set_title("Mean Test Accuracy — Original vs OASIS")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models_compare)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.4)

    # Above-chance accuracy (normalized)
    ax = axes[1]
    ax.bar(x - width/2, orig_above,  width, label="Original dataset", color="royalblue",  alpha=0.85)
    ax.bar(x + width/2, oasis_above, width, label="OASIS dataset",    color="darkorange", alpha=0.85)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, label="Chance level")
    ax.set_title("Accuracy Above Chance — Original vs OASIS")
    ax.set_ylabel("Accuracy − Chance (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models_compare)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.4)

    plt.suptitle("Cross-Dataset Generalization (Subject-Level, 5-Fold CV)", fontsize=13)
    plt.tight_layout()
    plt.savefig("models/cross_dataset_comparison.png", dpi=150)
    plt.show()
    print("Saved: models/cross_dataset_comparison.png")
else:
    print("Original kfold_results.json not found — skipping cross-dataset comparison.")


# --- 6. Summary table ---
print("\n" + "=" * 70)
print("OASIS K-FOLD SUMMARY (5-fold, subject-level majority vote)")
print("=" * 70)
header = f"\n{'Fold':<22} {'OasisCNN':>14} {'BrainResNet':>14}"
if use_rad:
    header += f" {'RadImageNet':>14}"
print(header)
print("-" * 70)
for i, (ca, ra) in enumerate(zip(cnn_fold_accs, resnet_fold_accs)):
    row = f"  Fold {i+1:<17} {ca:>13.2f}% {ra:>13.2f}%"
    if use_rad:
        row += f" {rad_fold_accs[i]:>13.2f}%"
    print(row)
print("-" * 70)
mean_row = f"{'Mean Accuracy':<22} {cnn_mean:>13.2f}% {resnet_mean:>13.2f}%"
std_row  = f"{'Std Dev':<22} {cnn_std:>13.2f}% {resnet_std:>13.2f}%"
if use_rad:
    mean_row += f" {rad_mean:>13.2f}%"
    std_row  += f" {rad_std:>13.2f}%"
print(mean_row)
print(std_row)
print(f"{'Chance baseline':<22} {'25.00%':>14} {'25.00%':>14}")

print("\n--- OasisCNN aggregated report ---")
print(classification_report(cnn_all_labels, cnn_all_preds, target_names=OASIS_CLASSES, zero_division=0))
print("--- BrainResNet aggregated report ---")
print(classification_report(resnet_all_labels, resnet_all_preds, target_names=OASIS_CLASSES, zero_division=0))
if use_rad:
    print("--- RadImageNet aggregated report ---")
    print(classification_report(rad_all_labels, rad_all_preds, target_names=OASIS_CLASSES, zero_division=0))
