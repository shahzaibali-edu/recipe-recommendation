import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

sys.path.append(str(Path(__file__).parent.parent / "src"))
from model import load_model
from dataloader import get_dataloaders

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"


# ── Run inference on entire test set ─────────────────────────────────────────

@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()

    all_preds  = []
    all_labels = []
    all_probs  = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1)
        preds   = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs)
    )


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, classes):
    cm = confusion_matrix(labels, preds)

    # Normalize to percentages per row
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(28, 22))

    # Raw counts
    sns.heatmap(
        cm, ax=axes[0],
        xticklabels=classes, yticklabels=classes,
        cmap="Blues", fmt="d", annot=True,
        annot_kws={"size": 7}, linewidths=0.3
    )
    axes[0].set_title("Confusion Matrix — Raw Counts", fontsize=13)
    axes[0].set_xlabel("Predicted", fontsize=11)
    axes[0].set_ylabel("Actual", fontsize=11)
    axes[0].tick_params(axis="x", rotation=90, labelsize=8)
    axes[0].tick_params(axis="y", rotation=0,  labelsize=8)

    # Normalized percentages
    sns.heatmap(
        cm_norm, ax=axes[1],
        xticklabels=classes, yticklabels=classes,
        cmap="Blues", fmt=".0f", annot=True,
        annot_kws={"size": 7}, linewidths=0.3,
        vmin=0, vmax=100
    )
    axes[1].set_title("Confusion Matrix — Normalized (%)", fontsize=13)
    axes[1].set_xlabel("Predicted", fontsize=11)
    axes[1].set_ylabel("Actual", fontsize=11)
    axes[1].tick_params(axis="x", rotation=90, labelsize=8)
    axes[1].tick_params(axis="y", rotation=0,  labelsize=8)

    plt.suptitle("Test Set Evaluation", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig("notebooks/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[Saved] confusion_matrix.png")

    return cm


# ── Per-class metrics bar chart ───────────────────────────────────────────────

def plot_per_class_accuracy(labels, preds, classes):
    cm      = confusion_matrix(labels, preds)
    per_cls = cm.diagonal() / cm.sum(axis=1) * 100

    # Sort by accuracy ascending so worst classes are obvious
    sorted_idx = np.argsort(per_cls)
    sorted_cls = [classes[i] for i in sorted_idx]
    sorted_acc = per_cls[sorted_idx]

    colors = ["#d9534f" if a < 80 else
              "#f0ad4e" if a < 90 else
              "#5cb85c" for a in sorted_acc]

    plt.figure(figsize=(14, 8))
    bars = plt.barh(sorted_cls, sorted_acc, color=colors, edgecolor="white")
    plt.axvline(x=90, color="gray", linestyle="--",
                linewidth=1, label="90% threshold")
    plt.axvline(x=np.mean(per_cls), color="blue", linestyle="--",
                linewidth=1, label=f"Mean: {np.mean(per_cls):.1f}%")

    # Add value labels on bars
    for bar, val in zip(bars, sorted_acc):
        plt.text(min(val + 0.5, 99), bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", fontsize=8)

    plt.xlabel("Accuracy (%)")
    plt.title("Per-class Accuracy on Test Set\n"
              "(red < 80%, orange 80–90%, green > 90%)")
    plt.xlim(0, 105)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("notebooks/per_class_accuracy.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[Saved] per_class_accuracy.png")

    return per_cls, sorted_cls, sorted_acc


# ── Top-3 accuracy ────────────────────────────────────────────────────────────

def top_k_accuracy(labels, probs, k=3):
    """
    Top-K accuracy: correct if true label is in top K predictions.
    More forgiving metric — useful when classes are visually similar.
    """
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct     = sum(
        labels[i] in top_k_preds[i]
        for i in range(len(labels))
    )
    return correct / len(labels) * 100


# ── Print full report ─────────────────────────────────────────────────────────

def print_report(labels, preds, probs, classes):
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION REPORT")
    print("=" * 60)

    top1 = accuracy_score(labels, preds) * 100
    top3 = top_k_accuracy(labels, probs, k=3)
    top5 = top_k_accuracy(labels, probs, k=5)

    print(f"\n  Top-1 Accuracy : {top1:.2f}%  (exact match)")
    print(f"  Top-3 Accuracy : {top3:.2f}%  (correct in top 3)")
    print(f"  Top-5 Accuracy : {top5:.2f}%  (correct in top 5)")
    print(f"\n  Total test images : {len(labels)}")
    print(f"  Correct (Top-1)   : {int(top1 * len(labels) / 100)}")
    print(f"  Wrong  (Top-1)    : {len(labels) - int(top1 * len(labels) / 100)}")

    print("\n" + "-" * 60)
    print("PER-CLASS REPORT (precision / recall / f1)")
    print("-" * 60)
    print(classification_report(labels, preds, target_names=classes, digits=3))


# ── Worst predictions visualizer ─────────────────────────────────────────────

def show_worst_predictions(loader, model, classes, device, n=10):
    """
    Show the n most confidently wrong predictions.
    These are the cases where the model was very sure but very wrong.
    Most revealing for understanding failure modes.
    """
    model.eval()
    wrong = []   # (confidence, true_label, pred_label, image_tensor)

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            probs   = torch.softmax(outputs, dim=1).cpu()
            preds   = probs.argmax(dim=1)

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    conf = probs[i][preds[i]].item()
                    wrong.append((conf, labels[i].item(),
                                  preds[i].item(), images[i]))

    # Sort by confidence descending — most confidently wrong first
    wrong.sort(key=lambda x: x[0], reverse=True)
    worst = wrong[:n]

    if not worst:
        print("\n[Perfect] No wrong predictions on test set!")
        return

    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    axes = axes.flatten()

    # Undo ImageNet normalization for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for idx, (conf, true_idx, pred_idx, img_t) in enumerate(worst):
        img = img_t * std + mean
        img = img.permute(1, 2, 0).clamp(0, 1).numpy()

        axes[idx].imshow(img)
        axes[idx].axis("off")
        axes[idx].set_title(
            f"True: {classes[true_idx]}\n"
            f"Pred: {classes[pred_idx]}\n"
            f"Conf: {conf:.1%}",
            fontsize=8,
            color="red"
        )

    # Hide unused subplots
    for idx in range(len(worst), len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Most Confidently Wrong Predictions", fontsize=12)
    plt.tight_layout()
    plt.savefig("notebooks/worst_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[Saved] worst_predictions.png")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # Load model
    model, classes = load_model(MODEL_PATH, DEVICE)
    print(f"\nLoaded model with {len(classes)} classes")

    # Load test data
    loaders = get_dataloaders()
    test_loader = loaders["test"]

    # Get all predictions
    print("\nRunning inference on test set...")
    preds, labels, probs = get_predictions(model, test_loader, DEVICE)

    # Print full report
    print_report(labels, preds, probs, classes)

    # Plots
    plot_confusion_matrix(labels, preds, classes)
    plot_per_class_accuracy(labels, preds, classes)
    show_worst_predictions(test_loader, model, classes, DEVICE, n=10)

    print("\nDone. Check notebooks/ for all saved plots.")