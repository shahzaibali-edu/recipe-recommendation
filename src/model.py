import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
NUM_CLASSES  = 36
DROPOUT_RATE = 0.3
MODELS_DIR   = Path("models")


# ── Model ─────────────────────────────────────────────────────────────────────

class IngredientClassifier(nn.Module):
    """
    EfficientNet-B0 backbone with a custom classification head
    for ingredient recognition.

    Architecture:
        EfficientNet-B0 backbone (pretrained, frozen initially)
            ↓
        Global Average Pooling  [built into backbone]
            ↓
        1280-dimensional feature vector
            ↓
        Dropout (0.3)
            ↓
        Linear (1280 → 256)
            ↓
        ReLU + Dropout (0.3)
            ↓
        Linear (256 → num_classes)
            ↓
        Raw logits (CrossEntropyLoss handles softmax)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = DROPOUT_RATE):
        super(IngredientClassifier, self).__init__()

        # ── Load pretrained EfficientNet-B0 ───────────────────────────────
        # Downloads ~20MB of pretrained weights on first run
        base_model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # ── Backbone — everything except the final classifier ─────────────
        # This is the pretrained part we freeze first
        self.backbone = base_model.features       # convolutional layers
        self.avgpool  = base_model.avgpool         # global average pooling → 1280-d

        # ── Our classification head — this is what we train ───────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

        # ── Freeze backbone by default ────────────────────────────────────
        self._freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        x shape in  : (batch_size, 3, 224, 224)
        x shape out : (batch_size, num_classes)
        """
        x = self.backbone(x)       # (B, 1280, 7, 7)
        x = self.avgpool(x)        # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)   # (B, 1280)
        x = self.classifier(x)    # (B, num_classes)
        return x

    # ── Freeze / Unfreeze helpers ─────────────────────────────────────────────

    def _freeze_backbone(self):
        """Freeze all backbone weights — only head trains."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[Model] Backbone frozen — training head only")

    def unfreeze_backbone(self, layers_from_end: int = 3):
        """
        Unfreeze the last N layers of the backbone for fine-tuning.
        We don't unfreeze everything — early layers detect basic edges
        and don't need to change. Later layers are more task-specific.

        Args:
            layers_from_end: how many backbone blocks to unfreeze (default 3)
        """
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Then unfreeze last N blocks
        backbone_layers = list(self.backbone.children())
        for layer in backbone_layers[-layers_from_end:]:
            for param in layer.parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Unfrozen last {layers_from_end} backbone layers")
        print(f"[Model] Trainable parameters: {trainable:,}")

    def freeze_backbone(self):
        """Re-freeze backbone — useful for switching back."""
        self._freeze_backbone()


# ── Utility functions ─────────────────────────────────────────────────────────

def get_model(num_classes: int = NUM_CLASSES) -> IngredientClassifier:
    """Build and return the model. Entry point for training script."""
    model = IngredientClassifier(num_classes=num_classes)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[Model] EfficientNet-B0 loaded")
    print(f"[Model] Total parameters     : {total:,}")
    print(f"[Model] Trainable parameters : {trainable:,}")

    return model


def save_model(model: IngredientClassifier, filename: str, classes: list):
    """
    Save model weights + class list together.
    We save classes so we can decode predictions later without
    needing the dataset present.
    """
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / filename

    torch.save({
        "model_state_dict" : model.state_dict(),
        "classes"          : classes,
        "num_classes"      : len(classes),
    }, path)

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"[Model] Saved → {path}  ({size_mb:.1f} MB)")


def load_model(filename: str, device: torch.device) -> tuple:
    """
    Load model weights and class list from a saved checkpoint.
    Returns (model, classes).
    """
    path      = MODELS_DIR / filename
    checkpoint = torch.load(path, map_location=device)

    model = IngredientClassifier(num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"[Model] Loaded ← {path}")
    print(f"[Model] Classes: {checkpoint['classes']}")

    return model, checkpoint["classes"]


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cpu")

    # Build model
    model = get_model()
    model.to(device)

    # Test forward pass with a fake batch
    dummy_input = torch.randn(4, 3, 224, 224)   # 4 images
    output      = model(dummy_input)

    print(f"\n[Test] Input shape  : {dummy_input.shape}")
    print(f"[Test] Output shape : {output.shape}")
    print(f"[Test] Expected     : torch.Size([4, 36])")
    assert output.shape == (4, NUM_CLASSES), "Output shape mismatch!"
    print("[Test] Forward pass OK")

    # Test unfreeze
    print("\n--- Testing unfreeze ---")
    model.unfreeze_backbone(layers_from_end=3)

    # Test save/load
    print("\n--- Testing save/load ---")
    dummy_classes = [f"class_{i}" for i in range(NUM_CLASSES)]
    save_model(model, "test_model.pth", dummy_classes)

    loaded_model, loaded_classes = load_model("test_model.pth", device)
    print(f"[Test] Loaded classes count: {len(loaded_classes)}")
    print("\n[OK] model.py fully verified")