import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import time
import json

from model import get_model, save_model
import sys
sys.path.append(str(Path(__file__).parent))
from dataloader import get_dataloaders

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS_HEAD    = 10      # Phase 1: train head only (backbone frozen)
EPOCHS_FINETUNE = 10     # Phase 2: fine-tune last backbone layers
LR_HEAD        = 1e-3    # higher LR when only training head
LR_FINETUNE    = 1e-4    # lower LR during fine-tuning
BATCH_SIZE     = 32
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH      = "best_model.pth"


# ── Training one epoch ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # ── Forward pass ──────────────────────────────────────────────────
        outputs = model(images)
        loss    = criterion(outputs, labels)

        # ── Backward pass ─────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── Metrics ───────────────────────────────────────────────────────
        preds          = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        total_loss    += loss.item()

        # Print progress every 20 batches
        if (batch_idx + 1) % 20 == 0:
            print(f"    Batch {batch_idx + 1}/{len(loader)} "
                  f"— Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples * 100
    return avg_loss, accuracy


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss    = criterion(outputs, labels)

        preds          = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        total_loss    += loss.item()

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples * 100
    return avg_loss, accuracy


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model, loaders, device):
    classes   = loaders["train"].dataset.classes
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    history   = {"train_loss": [], "val_loss": [],
                 "train_acc": [],  "val_acc": []}

    best_val_acc  = 0.0
    best_epoch    = 0

    # ── Phase 1: Train head only ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1 — Training classification head (backbone frozen)")
    print("=" * 60)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_HEAD)

    for epoch in range(1, EPOCHS_HEAD + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(
            model, loaders["validation"], criterion, device
        )
        scheduler.step()

        elapsed = time.time() - start

        print(f"\nEpoch {epoch:02d}/{EPOCHS_HEAD} ({elapsed:.1f}s) — "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            save_model(model, SAVE_PATH, classes)
            print(f"  ✓ New best model saved (val acc: {val_acc:.1f}%)")

    # ── Phase 2: Fine-tune last backbone layers ────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2 — Fine-tuning last 3 backbone layers")
    print("=" * 60)

    model.unfreeze_backbone(layers_from_end=3)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_FINETUNE, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_FINETUNE)

    for epoch in range(1, EPOCHS_FINETUNE + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(
            model, loaders["validation"], criterion, device
        )
        scheduler.step()

        elapsed = time.time() - start

        print(f"\nEpoch {epoch:02d}/{EPOCHS_FINETUNE} ({elapsed:.1f}s) — "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch + EPOCHS_HEAD
            save_model(model, SAVE_PATH, classes)
            print(f"  ✓ New best model saved (val acc: {val_acc:.1f}%)")

    # ── Save training history ──────────────────────────────────────────────
    history_path = Path("models/training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Best val accuracy : {best_val_acc:.1f}%  (epoch {best_epoch})")
    print(f"  Model saved to    : models/{SAVE_PATH}")
    print(f"  History saved to  : {history_path}")
    print("=" * 60)

    return history


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # Load data
    loaders = get_dataloaders(batch_size=BATCH_SIZE)

    # Build model
    model = get_model(num_classes=34)
    model = model.to(DEVICE)

    # Train
    history = train(model, loaders, DEVICE)