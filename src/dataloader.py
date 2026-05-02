import os
from pathlib import Path
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
BATCH_SIZE    = 32
NUM_WORKERS   = 0        # keep 0 on Windows to avoid multiprocessing issues
IMG_SIZE      = 224

# ImageNet mean and std — required for pretrained EfficientNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(split: str) -> transforms.Compose:
    """
    Returns the appropriate transform pipeline for each split.
    Train gets augmentations. Validation and test get clean transforms only.
    """

    if split == "train":
        return transforms.Compose([

       # ── Geometric ──────────────────────────────────────────────────
            transforms.RandomHorizontalFlip(p=0.5),

            # Pad → Rotate → Crop back — eliminates black corner triangles
            transforms.Pad(padding=30, fill=0, padding_mode="reflect"),
            transforms.RandomApply([
                transforms.RandomRotation(degrees=15)
            ], p=0.5),
            transforms.CenterCrop(IMG_SIZE),

            transforms.RandomResizedCrop(
                size=IMG_SIZE,
                scale=(0.80, 1.0),
                ratio=(0.90, 1.10),
            ),

        # ── Color ───────────────────────────────────────────────────────
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.15,                       # toned down from 0.2
            hue=0.02,                              # toned down from 0.05
        ),

        transforms.RandomGrayscale(p=0.05),

        # ── Convert & Normalize ─────────────────────────────────────────
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    else:
        # Validation and test — no augmentation, just clean resize and normalize
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class IngredientDataset(Dataset):
    """
    Loads ingredient images from the processed directory.
    Folder structure expected:
        data/processed/
            train/
                apple/
                    img1.jpg ...
                carrot/
                    img1.jpg ...
            validation/  ...
            test/        ...
    """

    def __init__(self, split: str, data_dir: Path = PROCESSED_DIR):
        self.split     = split
        self.data_dir  = data_dir / split
        self.transform = get_transforms(split)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Split folder not found: {self.data_dir}")

        # ── Build class list sorted alphabetically ──────────────────────────
        self.classes = sorted([
            d.name for d in self.data_dir.iterdir() if d.is_dir()
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # ── Build flat list of (image_path, label_index) tuples ─────────────
        self.samples = []
        for cls in self.classes:
            cls_dir = self.data_dir / cls
            for img_path in cls_dir.glob("*.*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append((img_path, self.class_to_idx[cls]))

        print(f"[{split:<10}] {len(self.samples)} images | {len(self.classes)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, label

    def get_class_name(self, idx: int) -> str:
        """Convert a label index back to a class name string."""
        return self.classes[idx]


# ── DataLoaders ───────────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir: Path = PROCESSED_DIR,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """
    Returns a dict with train, validation and test DataLoaders.
    Usage:
        loaders = get_dataloaders()
        for images, labels in loaders["train"]:
            ...
    """

    loaders = {}

    for split in ["train", "validation", "test"]:
        dataset = IngredientDataset(split=split, data_dir=data_dir)

        loaders[split] = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = (split == "train"),   # only shuffle training data
            num_workers = NUM_WORKERS,
            pin_memory  = False,                # set True if using GPU
        )

    return loaders


# ── Sanity check ──────────────────────────────────────────────────────────────

def verify_loaders(loaders: dict):
    """
    Pulls one batch from each split and prints shape info.
    Quick way to confirm everything is wired up correctly.
    """
    print("\n" + "=" * 50)
    print("DATALOADER VERIFICATION")
    print("=" * 50)

    for split, loader in loaders.items():
        images, labels = next(iter(loader))

        print(f"\n  [{split}]")
        print(f"    Batch image shape : {images.shape}")
        print(f"    Batch label shape : {labels.shape}")
        print(f"    Image dtype       : {images.dtype}")
        print(f"    Label dtype       : {labels.dtype}")
        print(f"    Pixel min/max     : {images.min():.3f} / {images.max():.3f}")
        print(f"    Sample labels     : {labels[:5].tolist()}")

        # decode label indices back to class names
        dataset    = loader.dataset
        class_names = [dataset.get_class_name(l.item()) for l in labels[:5]]
        print(f"    Sample classes    : {class_names}")

    print("\n[OK] All loaders working correctly")


def visualize_augmentations(data_dir: Path = PROCESSED_DIR, num_images: int = 6):
    """
    Shows the same image with different augmentations applied.
    Useful to visually confirm augmentations look realistic.
    """
    import matplotlib.pyplot as plt

    dataset  = IngredientDataset(split="train", data_dir=data_dir)
    inv_norm = transforms.Normalize(
        mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std =[1 / s for s in IMAGENET_STD]
    )

    # grab the first image raw (before transform)
    img_path, label = dataset.samples[0]
    class_name      = dataset.get_class_name(label)
    raw_img         = Image.open(img_path).convert("RGB")

    fig, axes = plt.subplots(2, num_images // 2, figsize=(14, 6))
    axes      = axes.flatten()

    transform = get_transforms("train")

    for i, ax in enumerate(axes):
        augmented = transform(raw_img)          # fresh random augmentation each time
        augmented = inv_norm(augmented)         # undo normalization for display
        augmented = augmented.permute(1, 2, 0)  # CHW → HWC for matplotlib
        augmented = augmented.clamp(0, 1)       # clip any out-of-range values

        ax.imshow(augmented.numpy())
        ax.axis("off")
        ax.set_title(f"Aug {i + 1}", fontsize=9)

    fig.suptitle(f"Augmentation samples — class: {class_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig("notebooks/augmentation_samples.png")
    plt.show()
    print("[Saved] augmentation_samples.png")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loaders = get_dataloaders()
    verify_loaders(loaders)
    visualize_augmentations()