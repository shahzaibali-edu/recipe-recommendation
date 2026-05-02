import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image, ImageStat
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SPLITS        = ["train", "validation", "test"]
TARGET_SIZE   = (224, 224)
MIN_ORIG_SIZE = 100        # images smaller than this (w or h) get removed
FLAG_LOG      = Path("notebooks/flagged_images.txt")

# Standardize folder names — fixes typos and inconsistencies in the dataset
NAME_MAP = {
    "raddish"        : "radish",
    "jalepeno"       : "jalapeno",
    "soy beans"      : "soybeans",
    "sweetcorn"      : "corn",         # fixed — go straight to corn
    "sweet_corn"     : "corn",         # fixed — handle both spellings
    "sweetpotato"    : "sweet_potato",
    "bell pepper"    : "bell_pepper",
    "chilli pepper"  : "chilli_pepper",
    "capsicum"       : "bell_pepper",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def is_likely_cartoon_or_logo(img, threshold=500):
    """Real photos have thousands of unique colors; cartoons/logos have very few."""
    small = img.resize((64, 64))
    pixels = np.array(small).reshape(-1, 3)
    unique_colors = len(set(map(tuple, pixels)))
    return unique_colors < threshold

def standardize_name(name: str) -> str:
    """Lowercase, strip spaces, apply name map."""
    cleaned = name.lower().strip()
    return NAME_MAP.get(cleaned, cleaned).replace(" ", "_")


def is_too_small(img: Image.Image) -> bool:
    """Flag images that are too small to be useful before resizing."""
    w, h = img.size
    return w < MIN_ORIG_SIZE or h < MIN_ORIG_SIZE


def is_too_dark_or_bright(img: Image.Image) -> bool:
    """
    Flag images with extreme brightness — likely black backgrounds or
    blown-out white studio shots that may confuse the model.
    Uses average pixel brightness across all channels.
    """
    stat    = ImageStat.Stat(img)
    avg_brightness = sum(stat.mean) / len(stat.mean)
    return avg_brightness < 20 or avg_brightness > 240


def has_low_contrast(img: Image.Image) -> bool:
    """
    Flag images with very low contrast — often watermarked infographics
    or near-solid color images. Uses standard deviation of pixel values.
    """
    stat   = ImageStat.Stat(img)
    avg_std = sum(stat.stddev) / len(stat.stddev)
    return avg_std < 15


def process_image(src_path: Path, dst_path: Path) -> dict:
    """
    Load, validate, resize, convert and save one image.
    Returns a result dict with status and optional flag reason.
    """
    result = {"src": str(src_path), "status": None, "flag": None}

    try:
        img = Image.open(src_path)

        # ── Check original size before anything ──
        if is_too_small(img):
            result["status"] = "removed"
            result["flag"]   = "too_small"
            return result

        # ── Convert to RGB (handles RGBA, grayscale, palette) ──
        img = img.convert("RGB")

        # ── Flag suspicious images but still keep them ──
        if is_too_dark_or_bright(img):
            result["flag"] = "extreme_brightness"
        elif has_low_contrast(img):
            result["flag"] = "low_contrast"
        if is_likely_cartoon_or_logo(img):
            result["status"] = "removed"
            result["flag"]   = "cartoon_or_logo"
            return result

        # ── Resize to target size ──
        img = img.resize(TARGET_SIZE, Image.LANCZOS)

        # ── Save to destination ──
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path, "JPEG", quality=95)

        result["status"] = "ok"

    except Exception as e:
        result["status"] = "error"
        result["flag"]   = str(e)

    return result


# ── Main pipeline ─────────────────────────────────────────────────────────────

def clean_dataset():
    stats = {
        "total"    : 0,
        "ok"       : 0,
        "removed"  : 0,
        "errors"   : 0,
        "flagged"  : 0,
    }

    flagged_entries  = []
    class_counts     = {}

    print("=" * 60)
    print("CLEANING DATASET")
    print(f"  Source : {RAW_DIR}")
    print(f"  Output : {PROCESSED_DIR}")
    print(f"  Target size: {TARGET_SIZE}")
    print("=" * 60)

    for split in SPLITS:
        split_src = RAW_DIR / split
        split_dst = PROCESSED_DIR / split

        if not split_src.exists():
            print(f"\n[SKIP] Split not found: {split_src}")
            continue

        print(f"\nProcessing split: {split}")

        class_folders = sorted([f for f in split_src.iterdir() if f.is_dir()])

        for class_folder in class_folders:
            std_name  = standardize_name(class_folder.name)
            dst_class = split_dst / std_name

            images = list(class_folder.glob("*.*"))

            ok_count = 0
            for img_path in images:
                # Skip non-image files
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png",
                                                    ".bmp", ".webp"]:
                    continue

                stats["total"] += 1

                dst_path = dst_class / (img_path.stem + ".jpg")
                result   = process_image(img_path, dst_path)

                if result["status"] == "ok":
                    stats["ok"] += 1
                    ok_count    += 1

                    if result["flag"]:
                        stats["flagged"] += 1
                        flagged_entries.append(
                            f"[{result['flag']}] {result['src']}"
                        )

                elif result["status"] == "removed":
                    stats["removed"] += 1
                    print(f"  [REMOVED] {img_path.name} — {result['flag']}")
                    flagged_entries.append(f"[{result['flag']}] {result['src']}")

                elif result["status"] == "error":
                    stats["errors"] += 1
                    print(f"  [ERROR]   {img_path.name} — {result['flag']}")

            # Track counts for summary
            if split == "train":
                class_counts[std_name] = ok_count

            print(f"  {std_name:<20} → {ok_count} images saved")

    return stats, flagged_entries, class_counts


def save_flag_log(flagged_entries: list):
    """Write flagged images to a text file for manual review."""
    FLAG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(FLAG_LOG, "w") as f:
        f.write(f"FLAGGED IMAGES — {len(flagged_entries)} total\n")
        f.write("These images were saved but may need manual review.\n")
        f.write("=" * 60 + "\n\n")
        for entry in flagged_entries:
            f.write(entry + "\n")

    print(f"\n[Saved] Flag log → {FLAG_LOG}")


def plot_class_counts(class_counts: dict):
    """Bar chart of training image counts after cleaning."""
    classes = list(class_counts.keys())
    counts  = list(class_counts.values())

    plt.figure(figsize=(14, 6))
    bars = plt.bar(classes, counts, color="steelblue", edgecolor="white")
    plt.axhline(y=np.mean(counts), color="red", linestyle="--",
                linewidth=1, label=f"Mean: {np.mean(counts):.0f}")
    plt.xticks(rotation=90, fontsize=9)
    plt.title("Training images per class after cleaning")
    plt.ylabel("Image count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("notebooks/cleaned_distribution.png")
    plt.show()
    print("[Saved] cleaned_distribution.png")


def print_final_summary(stats: dict, flagged_entries: list):
    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    print(f"  Total processed : {stats['total']}")
    print(f"  Saved (ok)      : {stats['ok']}")
    print(f"  Removed         : {stats['removed']}")
    print(f"  Errors          : {stats['errors']}")
    print(f"  Flagged         : {stats['flagged']}  ← review flagged_images.txt")
    print(f"\n  Processed data saved to: {PROCESSED_DIR}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    stats, flagged_entries, class_counts = clean_dataset()
    save_flag_log(flagged_entries)
    plot_class_counts(class_counts)
    print_final_summary(stats, flagged_entries)