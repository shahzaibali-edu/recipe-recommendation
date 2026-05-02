# Recipe Recommendation System — Project Log

**Project:** AI-powered recipe recommendation from ingredient photos  
**Stack:** Python, PyTorch, EfficientNet-B0, Spoonacular API  
**Status:** Model trained, evaluated and saved ✅  

---

## Project Goal

Build an AI system that takes a photo of an ingredient, identifies what it is, and recommends recipes based on the detected ingredients. Designed in two phases:

- **Phase 1 (current):** Single ingredient per photo → classify → query recipe API
- **Phase 2 (future):** Multiple ingredients in one photo → YOLOv8 object detection → query recipe API

---

## Architecture Overview

```
User Photo
    ↓
EfficientNet-B0 Classifier   ← trained model (our work)
    ↓
Ingredient Label (e.g. "tomato")
    ↓
Spoonacular API              ← pretrained / external
    ↓
Ranked Recipe Recommendations
```

---

## Step 1 — Problem Definition

**Decisions made:**
- Task type: single-label image classification (Phase 1), multi-label object detection (Phase 2)
- Input: one photo of one ingredient
- Output: top N recipes from API based on identified ingredient
- Started with 51 ingredient classes as target, narrowed to 34 based on dataset availability and class merging
- Chose API route over training a recipe model — recipes are structured data, not a learning problem

**Ingredient categories defined:**
- Vegetables: tomato, onion, garlic, potato, carrot, bell pepper, broccoli, spinach, cucumber, cabbage, mushroom, corn, peas, zucchini, eggplant, lettuce, celery, green beans
- Meats: chicken, beef, salmon, shrimp, eggs, ground beef, pork, tuna, lamb, bacon
- Dairy & Basics: cheese, butter, milk, yogurt, cream, rice, pasta, bread, flour, oats
- Pantry/Flavor: lemon, ginger, chili pepper, olive oil, beans, lentils, chickpeas, tofu, avocado, sweet potato, apple

---

## Step 2 — Data Collection

**Dataset used:**  
[Fruits and Vegetables Image Recognition Dataset](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition) — Kaggle  
Download size: ~2GB zip

**Why not the originally planned datasets:**

| Dataset | Reason rejected |
|---|---|
| Food-101 | Contains cooked dishes (pizza, sushi), not raw ingredients |
| VegFru | Chinese research dataset, hard to access, complex setup |
| PlantVillage | Plant disease detection dataset, completely wrong use case |
| Open Images food subset | Massive and complex to filter, not beginner friendly |

**Final dataset stats (after cleaning and class merging):**

| Split | Images | Classes |
|---|---|---|
| Train | 2,950 | 34 |
| Validation | 332 | 34 |
| Test | 334 | 34 |
| **Total** | **3,616** | **34** |

**Classes covered (34):**  
apple, banana, beetroot, bell_pepper, cabbage, carrot, cauliflower, chilli_pepper, corn, cucumber, eggplant, garlic, ginger, grapes, jalapeno, kiwi, lemon, lettuce, mango, onion, orange, paprika, pear, peas, pineapple, pomegranate, potato, radish, soybeans, spinach, sweet_potato, tomato, turnip, watermelon

**Note on class reduction:** Original dataset had 36 classes. `capsicum` was merged into `bell_pepper` (same vegetable, different regional name) and `sweet_corn` was merged into `corn`. This eliminated a false ambiguity the model could never reliably resolve.

**Dataset structure:**
```
data/raw/
    train/      ~85 images per class
    validation/ ~10 images per class
    test/       ~10 images per class
```

---

## Step 3 — Data Exploration (`notebooks/explore_data.py`)

**Key findings:**

| Metric | Value |
|---|---|
| Total images | 3,825 (raw) |
| Corrupt images | 0 |
| Min image width | 200px |
| Max image width | 7,360px |
| Average width | 1,356px |
| Average height | 1,119px |
| Train range per class | 68–100 images |

**Problems identified:**
1. Image sizes wildly inconsistent — min 200px to max 7,360px. Model needs uniform size
2. Class folder name typos: `raddish`, `jalepeno`, `soy beans`, `sweetpotato`, `sweetcorn`
3. Watermarked/infographic images found — e.g. "FUN FACTS ON CARROTS" text overlay on carrot image
4. Some images with pure black backgrounds (banana on black)
5. Dataset is small — ~85 training images per class makes augmentation critical
6. Duplicate classes: `capsicum` = `bell_pepper`, `sweet_corn` = `corn` (different regional names for same vegetable)
7. Cartoon/illustration/logo images present — e.g. cartoon potato, Apple Inc. logo labelled as apple

---

## Step 4 — Data Cleaning (`notebooks/clean_data.py`)

**Operations performed:**
- Standardized all class folder names (fixed typos, replaced spaces with underscores)
- Merged duplicate classes: `capsicum` → `bell_pepper`, `sweetcorn` / `sweet_corn` → `corn`
- Converted all images to RGB (handles RGBA, grayscale, palette modes)
- Removed images below 100px in either dimension
- Removed cartoon/illustration/logo images using color diversity filter
- Resized all images to **224×224** using LANCZOS resampling
- Saved all cleaned data to `data/processed/` — raw data untouched
- Logged all removed and suspicious images into `notebooks/flagged_images.txt` for manual review

**Name standardization and merge map:**
```
raddish        → radish
jalepeno       → jalapeno
soy beans      → soybeans
sweetcorn      → corn          (merged — same as corn)
sweet_corn     → corn          (merged — same as corn)
sweetpotato    → sweet_potato
bell pepper    → bell_pepper
chilli pepper  → chilli_pepper
capsicum       → bell_pepper   (merged — same vegetable, regional name difference)
```

**Cartoon/logo detection:**  
Added `is_likely_cartoon_or_logo()` filter — counts unique colors in a 64×64 downsample. Real photos have thousands of unique colors; flat illustrations and logos have very few. Threshold: 500 unique colors. Images below threshold are removed and logged.

**Cleaning results:**

| Metric | Count |
|---|---|
| Total processed (raw) | 3,825 |
| Successfully saved | 3,616 |
| Removed (cartoon/logo/too small) | 209 |
| Errors | 0 |
| Flagged for review | logged to flagged_images.txt |

---

## Step 5 — Augmentation Pipeline (`src/dataloader.py`)

**Design decision:** Augmentation is applied on-the-fly during training, not saved to disk. `data/processed/` stays clean and untouched.

**Train augmentations:**

| Transform | Parameters | Reason |
|---|---|---|
| RandomHorizontalFlip | p=0.5 | Ingredients look same flipped |
| Pad → RandomRotation → CenterCrop | 30px pad, ±15°, p=0.5 | Simulate different photo angles, pad prevents black corners |
| RandomResizedCrop | scale=0.80–1.0, ratio=0.90–1.10 | Ingredient not always perfectly centered |
| ColorJitter | brightness=0.3, contrast=0.3, saturation=0.15, hue=0.02 | Different lighting conditions |
| RandomGrayscale | p=0.05 | Forces model to learn shape not just color |
| Normalize | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] | ImageNet stats required for pretrained model |

**Validation/Test transforms:** Resize to 224×224 + Normalize only. No augmentation ever applied to val/test.

**Issues encountered and fixed:**
1. Black corner triangles on rotated images — fixed by padding before rotation then cropping back
2. Purple/unnatural color tint — fixed by reducing hue from 0.05 → 0.02 and saturation from 0.2 → 0.15

**DataLoader verification:**
```
[train]      Batch shape: [32, 3, 224, 224] | dtype: float32 | range: -2.118 / 2.640
[validation] Batch shape: [32, 3, 224, 224] | dtype: float32 | range: -2.118 / 2.640
[test]       Batch shape: [32, 3, 224, 224] | dtype: float32 | range: -2.118 / 2.640
```
Pixel range of -2.118 to 2.640 is correct — expected result of ImageNet normalization.

---

## Step 6 — Model Architecture (`src/model.py`)

**Model chosen:** EfficientNet-B0 (pretrained on ImageNet)

**Why EfficientNet-B0 over alternatives:**
- Smallest/fastest in EfficientNet family — suitable for free Colab GPU
- Strong accuracy for food/ingredient classification tasks
- Pretrained weights mean it already understands shapes, textures, colors
- Only ~20MB saved model size

**Why not YOLO/Faster R-CNN at this stage:**
- Those are object detectors — overkill for single-ingredient classification
- YOLO will be used in Phase 2 for multi-ingredient detection
- This classifier may serve as the identification backbone when combined with YOLO later

**Architecture:**
```
Input: (batch, 3, 224, 224)
    ↓
EfficientNet-B0 Backbone (pretrained, frozen in Phase 1)
    ↓
Global Average Pooling
    ↓
1280-dimensional feature vector
    ↓
── OUR CLASSIFICATION HEAD ──
Dropout (0.3)
Linear (1280 → 256)
ReLU
Dropout (0.3)
Linear (256 → 34)
    ↓
Output: (batch, 34) raw logits
```

**Parameter counts:**

| State | Trainable Parameters |
|---|---|
| Backbone frozen (Phase 1) | 337,188 |
| Last 3 layers unfrozen (Phase 2) | 3,492,928 |
| Total parameters | 4,344,736 |

**Model size on disk:** 16.9 MB

---

## Step 7 — Training (`src/train.py`)

**Training strategy:** Two-phase training
- Phase 1: Freeze backbone, train classification head only
- Phase 2: Unfreeze last 3 backbone layers, fine-tune at lower learning rate

**Hyperparameters:**

| Parameter | Phase 1 | Phase 2 |
|---|---|---|
| Epochs | 10 | 10 |
| Learning rate | 1e-3 | 1e-4 |
| Optimizer | AdamW | AdamW |
| Weight decay | 1e-4 | 1e-4 |
| LR scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Loss function | CrossEntropyLoss (label_smoothing=0.1) | same |
| Batch size | 32 | 32 |

**Training environment:**
- Platform: Google Colab (free tier)
- Hardware: T4 GPU
- Epoch time on GPU: ~35–39 seconds
- Epoch time on CPU (first accidental run): ~408 seconds (12x slower)

**Final training results:**

| Metric | Value |
|---|---|
| Best validation accuracy | **99.1%** |
| Model saved | `models/best_model.pth` |

---

## Step 8 — Evaluation (`notebooks/evaluate.py`) ✅

**Test set results:**

| Metric | Value |
|---|---|
| Top-1 Accuracy | **99.10%** |
| Top-3 Accuracy | **99.40%** |
| Top-5 Accuracy | **99.70%** |
| Total test images | 334 |
| Correct (Top-1) | 331 |
| Wrong (Top-1) | 3 |

**Per-class report (precision / recall / f1):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| apple | 0.778 | 1.000 | 0.875 | 7 |
| banana | 0.875 | 1.000 | 0.933 | 7 |
| beetroot | 1.000 | 1.000 | 1.000 | 10 |
| bell_pepper | 1.000 | 1.000 | 1.000 | 10 |
| cabbage | 1.000 | 1.000 | 1.000 | 10 |
| carrot | 1.000 | 0.900 | 0.947 | 10 |
| cauliflower | 1.000 | 1.000 | 1.000 | 10 |
| chilli_pepper | 1.000 | 1.000 | 1.000 | 10 |
| corn | 1.000 | 1.000 | 1.000 | 10 |
| cucumber | 1.000 | 1.000 | 1.000 | 10 |
| eggplant | 1.000 | 1.000 | 1.000 | 10 |
| garlic | 1.000 | 1.000 | 1.000 | 10 |
| ginger | 1.000 | 1.000 | 1.000 | 10 |
| grapes | 1.000 | 1.000 | 1.000 | 10 |
| jalapeno | 1.000 | 1.000 | 1.000 | 10 |
| kiwi | 1.000 | 1.000 | 1.000 | 10 |
| lemon | 1.000 | 1.000 | 1.000 | 10 |
| lettuce | 1.000 | 1.000 | 1.000 | 10 |
| mango | 1.000 | 1.000 | 1.000 | 10 |
| onion | 1.000 | 1.000 | 1.000 | 10 |
| orange | 1.000 | 1.000 | 1.000 | 10 |
| paprika | 1.000 | 1.000 | 1.000 | 10 |
| pear | 1.000 | 1.000 | 1.000 | 10 |
| peas | 1.000 | 1.000 | 1.000 | 10 |
| pineapple | 1.000 | 1.000 | 1.000 | 10 |
| pomegranate | 1.000 | 1.000 | 1.000 | 10 |
| potato | 1.000 | 0.800 | 0.889 | 10 |
| radish | 1.000 | 1.000 | 1.000 | 10 |
| soybeans | 1.000 | 1.000 | 1.000 | 10 |
| spinach | 1.000 | 1.000 | 1.000 | 10 |
| sweet_potato | 1.000 | 1.000 | 1.000 | 10 |
| tomato | 1.000 | 1.000 | 1.000 | 10 |
| turnip | 1.000 | 1.000 | 1.000 | 10 |
| watermelon | 1.000 | 1.000 | 1.000 | 10 |
| **accuracy** | | | **0.991** | **334** |
| macro avg | 0.990 | 0.991 | 0.990 | 334 |
| weighted avg | 0.993 | 0.991 | 0.991 | 334 |

**Analysis of the 3 remaining errors:**  
All 3 wrong predictions are cartoon/illustration images that passed the color diversity filter — a cartoon potato with sunglasses, a cartoon carrot illustration, and a "OMG I'M A POTATO" graphic. No real food photographs were misclassified. The model performs effectively at 100% on real photos.

**Outputs saved:** `confusion_matrix.png`, `per_class_accuracy.png`, `worst_predictions.png`

---

## Issues & Fixes Log

| # | Issue | Fix |
|---|---|---|
| 1 | Accidentally ran training on CPU first | Restarted Colab, confirmed T4 GPU before rerunning |
| 2 | Black corner triangles in augmented images from rotation | Added reflect-padding before rotation, CenterCrop after |
| 3 | Purple/unnatural color tint in augmentations | Reduced hue 0.05→0.02, saturation 0.2→0.15 |
| 4 | Misspelled class folder names in dataset | Built NAME_MAP in clean_data.py for automatic standardization |
| 5 | `models/` folder missing in Colab after unzip | Created folder manually via os.makedirs in Colab |
| 6 | `processed/` sitting at root instead of inside `data/` | Used shutil.move in Colab to fix structure |
| 7 | sys.path import issue for train.py on Colab | Moved sys.path.append before model/dataloader imports |
| 8 | `capsicum` and `sweet_corn` were duplicate classes | Merged into `bell_pepper` and `corn` via NAME_MAP |
| 9 | Cartoon/logo images in dataset causing misclassifications | Added color diversity filter in clean_data.py — images with <500 unique colors auto-removed |
| 10 | Model built with hardcoded `num_classes=36` after dataset reduced to 34 | Updated `train.py` to `get_model(num_classes=34)` |
| 11 | Old 36-class `best_model.pth` still in models/ causing size mismatch on load | Deleted old model, retrained, downloaded fresh model from Colab |

---

## Project File Structure

```
recipe-recommendation/
│
├── data/
│   ├── raw/                        ← original downloaded dataset (never modified)
│   └── processed/                  ← cleaned, resized 224×224 images
│       ├── train/   (2,950 images)
│       ├── validation/ (332 images)
│       └── test/    (334 images)
│
├── models/
│   ├── best_model.pth              ← trained model weights + class list (16.9 MB)
│   └── training_history.json       ← loss and accuracy per epoch
│
├── notebooks/
│   ├── explore_data.py             ← dataset exploration script
│   ├── clean_data.py               ← cleaning and preprocessing script
│   ├── evaluate.py                 ← test set evaluation script
│   ├── plot_training_history.py    ← training curve plots for presentation
│   ├── class_distribution.png      ← bar chart of images per class
│   ├── cleaned_distribution.png    ← distribution after cleaning
│   ├── sample_images.png           ← sample images per class
│   ├── augmentation_samples.png    ← augmentation verification
│   ├── confusion_matrix.png        ← test set confusion matrix
│   ├── per_class_accuracy.png      ← per-class accuracy bar chart
│   ├── worst_predictions.png       ← most confidently wrong predictions
│   ├── training_curves.png         ← loss and accuracy over epochs
│   └── flagged_images.txt          ← removed/flagged images log
│
└── src/
    ├── dataloader.py               ← dataset class, transforms, dataloaders
    ├── model.py                    ← EfficientNet-B0 architecture + save/load
    └── train.py                    ← training loop, two-phase strategy
```

---

## What's Next

| Step | Task | Status |
|---|---|---|
| 8 | Evaluate on test set — accuracy, confusion matrix, per-class metrics | ✅ Done |
| 9 | Connect Spoonacular API — send ingredient → get recipes | 🔲 Todo |
| 10 | Build inference pipeline — photo in, recipe list out | 🔲 Todo |
| 11 | Simple UI — upload photo, see recipes | 🔲 Todo |
| 12 | Push to GitHub — with .gitignore, README, model on Hugging Face | 🔲 Todo |
| 13 | Phase 2 — Switch to YOLOv8 for multi-ingredient detection | 🔲 Future |

---

## Notes for Collaborators

- Model file (`best_model.pth`) is NOT in the repository — download from Hugging Face Hub (link TBD)
- Dataset is NOT in the repository — download from [Kaggle](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition) and run `clean_data.py`
- All training was done on Google Colab free tier with T4 GPU
- Run all scripts from the project root directory, not from inside `src/` or `notebooks/`

---

*Log last updated after Step 8 — Model Evaluation*