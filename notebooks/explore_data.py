import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ── Config ───────────────────────────────────────────────
DATA_DIR = Path("data/raw")
SPLITS    = ["train", "test", "validation"]

# ── 1. Count images per class per split ──────────────────
def count_images(data_dir):
    summary = {}

    for split in SPLITS:
        split_path = data_dir / split
        if not split_path.exists():
            print(f"[WARNING] Split folder not found: {split_path}")
            continue

        summary[split] = {}
        for class_folder in sorted(split_path.iterdir()):
            if class_folder.is_dir():
                count = len(list(class_folder.glob("*.*")))
                summary[split][class_folder.name] = count

    return summary


# ── 2. Print summary table ────────────────────────────────
def print_summary(summary):
    print(f"\n{'Class':<20} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-" * 50)

    all_classes = sorted(summary.get("train", {}).keys())

    for cls in all_classes:
        train = summary.get("train",      {}).get(cls, 0)
        val   = summary.get("validation", {}).get(cls, 0)
        test  = summary.get("test",       {}).get(cls, 0)
        total = train + val + test
        print(f"{cls:<20} {train:>8} {val:>8} {test:>8} {total:>8}")

    print("-" * 50)
    total_train = sum(summary.get("train",      {}).values())
    total_val   = sum(summary.get("validation", {}).values())
    total_test  = sum(summary.get("test",       {}).values())
    print(f"{'TOTAL':<20} {total_train:>8} {total_val:>8} {total_test:>8} {total_train+total_val+total_test:>8}")


# ── 3. Plot class distribution ────────────────────────────
def plot_distribution(summary):
    classes = sorted(summary.get("train", {}).keys())
    counts  = [summary["train"].get(cls, 0) for cls in classes]

    plt.figure(figsize=(14, 6))
    sns.barplot(x=classes, y=counts, palette="viridis")
    plt.xticks(rotation=90)
    plt.title("Training images per class")
    plt.ylabel("Image count")
    plt.tight_layout()
    plt.savefig("notebooks/class_distribution.png")
    plt.show()
    print("\n[Saved] class_distribution.png")


# ── 4. Show sample images per class ──────────────────────
def show_samples(data_dir, num_classes=8, samples_per_class=3):
    train_path = data_dir / "train"
    classes    = sorted([f.name for f in train_path.iterdir() if f.is_dir()])[:num_classes]

    fig, axes = plt.subplots(num_classes, samples_per_class,
                             figsize=(samples_per_class * 3, num_classes * 3))

    for row, cls in enumerate(classes):
        images = list((train_path / cls).glob("*.*"))[:samples_per_class]
        for col, img_path in enumerate(images):
            try:
                img = Image.open(img_path).convert("RGB")
                axes[row, col].imshow(img)
                axes[row, col].axis("off")
                if col == 0:
                    axes[row, col].set_ylabel(cls, fontsize=9, rotation=0,
                                              labelpad=60, va="center")
            except Exception as e:
                print(f"[ERROR] Could not open {img_path}: {e}")

    plt.suptitle("Sample images per class", fontsize=13)
    plt.tight_layout()
    plt.savefig("notebooks/sample_images.png")
    plt.show()
    print("[Saved] sample_images.png")


# ── 5. Check for corrupt images ──────────────────────────
def check_corrupt(data_dir):
    corrupt = []

    for split in SPLITS:
        split_path = data_dir / split
        if not split_path.exists():
            continue
        for img_path in split_path.rglob("*.*"):
            try:
                img = Image.open(img_path)
                img.verify()
            except Exception:
                corrupt.append(str(img_path))

    if corrupt:
        print(f"\n[WARNING] Found {len(corrupt)} corrupt images:")
        for p in corrupt:
            print(f"  {p}")
    else:
        print("\n[OK] No corrupt images found")

    return corrupt


# ── 6. Check image sizes ──────────────────────────────────
def check_sizes(data_dir, sample_limit=200):
    sizes = []
    count = 0

    for img_path in (data_dir / "train").rglob("*.*"):
        if count >= sample_limit:
            break
        try:
            img = Image.open(img_path)
            sizes.append(img.size)
            count += 1
        except:
            pass

    widths  = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]

    print(f"\nImage size stats (sampled {count} images from train):")
    print(f"  Width  — min: {min(widths)}, max: {max(widths)}, avg: {int(np.mean(widths))}")
    print(f"  Height — min: {min(heights)}, max: {max(heights)}, avg: {int(np.mean(heights))}")


# ── Run everything ────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("DATASET EXPLORATION")
    print("=" * 50)

    summary = count_images(DATA_DIR)
    print_summary(summary)
    plot_distribution(summary)
    show_samples(DATA_DIR)
    check_corrupt(DATA_DIR)
    check_sizes(DATA_DIR)

    print("\nDone. Check notebooks/ for saved plots.")