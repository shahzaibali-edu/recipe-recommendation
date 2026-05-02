import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Load history ──────────────────────────────────────────────────────────────
with open("models/training_history.json", "r") as f:
    history = json.load(f)

train_loss = history["train_loss"]
val_loss   = history["val_loss"]
train_acc  = history["train_acc"]
val_acc    = history["val_acc"]

epochs      = list(range(1, len(train_loss) + 1))
phase_split = 10   # Phase 1 = epochs 1-10, Phase 2 = epochs 11-20

# ── Plot config ───────────────────────────────────────────────────────────────
plt.rcParams["font.family"]  = "DejaVu Sans"
plt.rcParams["font.size"]    = 11
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("EfficientNet-B0 Training — Recipe Ingredient Classifier",
             fontsize=14, fontweight="bold", y=1.02)

PHASE_COLOR = "#f0f0f0"
P1_LABEL    = "Phase 1\n(head only)"
P2_LABEL    = "Phase 2\n(fine-tune)"

# ── Loss plot ─────────────────────────────────────────────────────────────────
ax1.axvspan(1, phase_split + 0.5, color=PHASE_COLOR, zorder=0)
ax1.axvspan(phase_split + 0.5, len(epochs),  color="white",      zorder=0)
ax1.axvline(phase_split + 0.5, color="#cccccc", linestyle="--", linewidth=1)

ax1.plot(epochs, train_loss, color="#2563eb", linewidth=2,
         marker="o", markersize=4, label="Train Loss")
ax1.plot(epochs, val_loss,   color="#16a34a", linewidth=2,
         marker="o", markersize=4, label="Val Loss")

ax1.set_title("Loss over Epochs", fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

# Phase labels
ax1.text(phase_split / 2 + 0.5, ax1.get_ylim()[1] if ax1.get_ylim()[1] else max(train_loss), P1_LABEL, ha="center", va="top", fontsize=9, color="#888888")

# ── Accuracy plot ─────────────────────────────────────────────────────────────
ax2.axvspan(1, phase_split + 0.5, color=PHASE_COLOR, zorder=0)
ax2.axvspan(phase_split + 0.5, len(epochs),  color="white",      zorder=0)
ax2.axvline(phase_split + 0.5, color="#cccccc", linestyle="--", linewidth=1)

ax2.plot(epochs, train_acc, color="#2563eb", linewidth=2,
         marker="o", markersize=4, label="Train Acc")
ax2.plot(epochs, val_acc,   color="#16a34a", linewidth=2,
         marker="o", markersize=4, label="Val Acc")

# Mark best val accuracy
best_epoch = int(np.argmax(val_acc)) + 1
best_acc   = max(val_acc)
ax2.annotate(f"Best: {best_acc:.1f}%\n(epoch {best_epoch})",
             xy=(best_epoch, best_acc),
             xytext=(best_epoch - 3, best_acc - 6),
             arrowprops=dict(arrowstyle="->", color="#dc2626"),
             fontsize=9, color="#dc2626")

ax2.set_title("Accuracy over Epochs (%)", fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_ylim(30, 105)
ax2.legend()

# Phase labels
ax2.text(phase_split / 2 + 0.5, 35, P1_LABEL,
         ha="center", fontsize=9, color="#888888")
ax2.text(phase_split + (len(epochs) - phase_split) / 2 + 0.5, 35, P2_LABEL,
         ha="center", fontsize=9, color="#888888")

# ── Final stats box ───────────────────────────────────────────────────────────
stats_text = (
    f"Best Val Acc : {best_acc:.1f}%  (epoch {best_epoch})\n"
    f"Final Train Acc : {train_acc[-1]:.1f}%\n"
    f"Total Epochs : {len(epochs)}  (10 + 10)"
)
fig.text(0.5, -0.04, stats_text, ha="center", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc",
                   edgecolor="#e2e8f0"))

plt.tight_layout()
plt.savefig("notebooks/training_curves.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.show()
print("[Saved] notebooks/training_curves.png")