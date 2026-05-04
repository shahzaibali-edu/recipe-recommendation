"""
app.py — AI Recipe Recommender (Multi-Ingredient Edition)
==========================================================
- Upload MULTIPLE ingredient images at once
- Each image is run through best_model.pth independently
- All detected ingredients are combined and sent to Spoonacular as ONE query
- Recipes that use the most of your ingredients are returned
"""

import os
import sys
import hashlib
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# ── Make src/ importable ───────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Load API Key ───────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("SPOONACULAR_API_KEY", "").strip()

# ── Model path ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pth")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING — cached so it loads once and stays in memory
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ingredient model...")
def load_our_model():
    """
    Loads best_model.pth (IngredientClassifier — EfficientNet-B0, 36 classes).
    Returns (model, classes, transform, device) or (None,None,None,None).
    """
    try:
        import torch
        from torchvision import transforms
        from src.model import IngredientClassifier

        device     = torch.device("cpu")
        checkpoint = torch.load(MODEL_PATH, map_location=device)

        model = IngredientClassifier(num_classes=checkpoint["num_classes"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])

        return model, checkpoint["classes"], transform, device

    except Exception as e:
        return None, None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def file_hash(f) -> str:
    """MD5 of file bytes — used to detect when a new set of files is uploaded."""
    f.seek(0); data = f.read(); f.seek(0)
    return hashlib.md5(data).hexdigest()


def predict_ingredient(image: Image.Image) -> tuple[str | None, float]:
    """
    Run best_model.pth on one PIL image.
    Returns (class_name, confidence) or (None, 0.0).
    """
    model, classes, transform, device = load_our_model()
    if model is None:
        return None, 0.0
    try:
        import torch
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)
        top_prob, top_idx = probs.max(dim=1)
        return classes[top_idx.item()], top_prob.item()
    except Exception:
        return None, 0.0


def get_recipes(ingredients_csv: str, n: int = 6) -> list | str:
    """
    Query Spoonacular findByIngredients with a comma-separated ingredient string.
    Returns list of recipe dicts, "quota" on 402, or [] on error.
    """
    if not API_KEY:
        return []
    try:
        resp = requests.get(
            "https://api.spoonacular.com/recipes/findByIngredients",
            params={
                "ingredients": ingredients_csv,
                "number":      n,
                "ranking":     2,          # maximise used ingredients
                "ignorePantry": True,
                "apiKey":      API_KEY,
            },
            timeout=10,
        )
        if resp.status_code == 200:   return resp.json()
        if resp.status_code == 402:   return "quota"
        return []
    except requests.exceptions.RequestException:
        return []


def compute_batch_hash(files) -> str:
    """Single hash representing the entire set of uploaded files."""
    combined = b"".join(
        (f.seek(0) or f.read()) for f in files
    )
    for f in files:
        f.seek(0)
    return hashlib.md5(combined).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Recipe AI — Multi-Ingredient Detector",
    page_icon="🍽️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}
header[data-testid="stHeader"] { background: transparent; }

/* Hero */
.hero { text-align:center; padding:2.5rem 1rem 1.5rem; }
.hero-title {
    font-size:3rem; font-weight:700;
    background: linear-gradient(90deg,#f7971e,#ffd200,#f7971e);
    background-size:200%;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    animation: shimmer 3s infinite linear;
}
@keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }
.hero-sub { font-size:1rem; color:rgba(255,255,255,0.5); }

/* Ingredient detection cards (thumbnail grid) */
.detect-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 0.8rem;
    text-align: center;
    transition: border-color 0.2s;
}
.detect-card:hover { border-color: rgba(247,151,30,0.4); }
.detect-name {
    font-size:0.88rem; font-weight:600;
    color:#ffd200; margin-top:0.4rem;
}
.detect-conf { font-size:0.75rem; color:rgba(255,255,255,0.4); }

/* Ingredient summary pill */
.ingr-pill {
    display:inline-block;
    background: linear-gradient(135deg,#11998e,#38ef7d);
    color:white; font-weight:600; font-size:0.85rem;
    padding:4px 14px; border-radius:20px; margin:3px;
}
.ingr-pill-manual {
    display:inline-block;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    color:rgba(255,255,255,0.7); font-size:0.85rem;
    padding:4px 14px; border-radius:20px; margin:3px;
}

/* Recipe cards */
.recipe-card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius:20px; padding:1.4rem 1.6rem;
    margin-bottom:1.2rem;
    transition: transform 0.2s, border-color 0.2s;
}
.recipe-card:hover { transform:translateY(-4px); border-color:rgba(247,151,30,0.5); }
.recipe-title { font-size:1.15rem; font-weight:600; color:#ffd200; margin-bottom:.3rem; }
.recipe-meta  { font-size:0.82rem; color:rgba(255,255,255,0.45); }
.miss-pill {
    display:inline-block;
    background:rgba(247,151,30,0.12);
    border:1px solid rgba(247,151,30,0.25);
    color:#ffd200; border-radius:20px;
    padding:2px 10px; font-size:0.78rem; margin:2px;
}

/* Section labels */
.section-label {
    font-size:0.7rem; font-weight:600;
    letter-spacing:2px; text-transform:uppercase;
    color:rgba(255,255,255,0.3); margin-bottom:0.5rem;
}

/* Status */
.status-ok  { color:#38ef7d; font-size:0.8rem; }
.status-bad { color:#ff6b6b; font-size:0.8rem; }

hr { border-color:rgba(255,255,255,0.07) !important; }

/* Buttons */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg,#f7971e,#ffd200);
    color:#1a1a2e; font-weight:700; font-size:1rem;
    border:none; border-radius:12px;
    padding:.6rem 2rem; width:100%;
    transition: opacity .2s, transform .15s;
}
div[data-testid="stButton"] > button:hover { opacity:.85; transform:translateY(-2px); }

/* Text input */
div[data-testid="stTextInput"] input {
    background:rgba(255,255,255,0.06) !important;
    border:1px solid rgba(255,255,255,0.15) !important;
    border-radius:10px !important; color:white !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for key, default in {
    "batch_hash":         None,
    "detections":         [],   # list of {filename, image, name, confidence}
    "final_ingredients":  [],   # merged & deduplicated list of ingredient strings
    "recipes":            [],
    "analyzed":           False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🍽️ AI Recipe Recommender</div>
    <div class="hero-sub">Upload multiple ingredient photos · AI detects each one · Get recipes using all of them</div>
</div>
""", unsafe_allow_html=True)

# Status bar
_, col_model, col_api = st.columns([3, 1.8, 1.4])
with col_model:
    st.markdown('<p class="status-ok" style="text-align:center">🧠 Model: 36 ingredients · 99% acc</p>', unsafe_allow_html=True)
with col_api:
    if API_KEY:
        st.markdown('<p class="status-ok" style="text-align:right">🟢 API Connected</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-bad" style="text-align:right">🔴 API Key Missing</p>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — FILE UPLOAD (multiple)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">📸 Step 1 — Upload Ingredient Images (select multiple)</p>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    label="upload",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if not uploaded_files:
    st.markdown("""
    <div style="background:rgba(255,255,255,0.03);border:2px dashed rgba(247,151,30,0.3);
                border-radius:20px;padding:2.5rem;text-align:center;margin:0.5rem 0 1rem">
        <p style="font-size:2.5rem;margin:0">📷</p>
        <p style="color:rgba(255,255,255,0.45);margin:.5rem 0 0">
            Click above to select <b style="color:#ffd200">one or more</b> ingredient photos<br>
            <small>The AI will classify each image separately and combine them for recipe search</small>
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    # ── Detect if new files were uploaded ─────────────────────────────────────
    current_batch_hash = compute_batch_hash(uploaded_files)
    if current_batch_hash != st.session_state.batch_hash:
        # New upload batch — run inference on every file
        st.session_state.batch_hash = current_batch_hash
        st.session_state.recipes    = []
        st.session_state.analyzed   = False

        detections = []
        with st.spinner(f"Running AI on {len(uploaded_files)} image(s)..."):
            for f in uploaded_files:
                img  = Image.open(f).convert("RGB")
                name, conf = predict_ingredient(img)
                detections.append({
                    "filename":   f.name,
                    "image":      img,
                    "name":       name,
                    "confidence": conf,
                })

        st.session_state.detections = detections
        # Build initial ingredient list from detections
        st.session_state.final_ingredients = [
            d["name"].replace("_", " ") for d in detections if d["name"]
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — Show detection results grid
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">🤖 Step 2 — AI Detection Results</p>', unsafe_allow_html=True)

    n_imgs = len(st.session_state.detections)
    cols_per_row = min(n_imgs, 4)
    grid_cols = st.columns(cols_per_row)

    for i, det in enumerate(st.session_state.detections):
        with grid_cols[i % cols_per_row]:
            st.image(det["image"], use_container_width=True)
            if det["name"]:
                st.markdown(
                    f'<div class="detect-name">🤖 {det["name"].replace("_"," ").capitalize()}</div>'
                    f'<div class="detect-conf">{det["confidence"]*100:.1f}% confident</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="detect-conf" style="color:#ff6b6b">❓ Not recognised</div>',
                    unsafe_allow_html=True,
                )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — Review & edit the combined ingredient list
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">✏️ Step 3 — Review & Edit Ingredients (add/remove as needed)</p>', unsafe_allow_html=True)

    # Show current detected pills
    detected_pills = "".join(
        f'<span class="ingr-pill">✅ {n}</span>'
        for n in st.session_state.final_ingredients
    )
    if detected_pills:
        st.markdown(f"<div style='margin-bottom:.6rem'>{detected_pills}</div>", unsafe_allow_html=True)

    # Editable text field — pre-filled with detected names
    ingredient_str = st.text_input(
        label="ingredients_edit",
        value=", ".join(st.session_state.final_ingredients),
        placeholder="e.g. tomato, garlic, chicken, onion",
        label_visibility="collapsed",
        help="Add or remove ingredients, separate with commas",
    )

    # Sync session state
    edited = [i.strip() for i in ingredient_str.split(",") if i.strip()]
    st.session_state.final_ingredients = edited

    if not edited:
        st.warning("No ingredients detected or entered. Please type at least one ingredient.")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — Find Recipes button
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">🍳 Step 4 — Find Matching Recipes</p>', unsafe_allow_html=True)

    if edited:
        ingr_csv = ", ".join(edited)
        st.markdown(
            f"<p style='color:rgba(255,255,255,0.4);font-size:0.88rem;margin-bottom:.8rem'>"
            f"Will search for recipes using: "
            f"<b style='color:white'>{ingr_csv}</b></p>",
            unsafe_allow_html=True,
        )

    if st.button("🔍 Find Recipes for All Ingredients", use_container_width=True):
        if not edited:
            st.error("Please add at least one ingredient.")
        elif not API_KEY:
            st.error("❌ SPOONACULAR_API_KEY is missing in your .env file.")
        else:
            with st.spinner("Querying Spoonacular with all your ingredients..."):
                result = get_recipes(", ".join(edited))

            if result == "quota":
                st.warning("⚠️ Spoonacular API daily quota exceeded. Try again tomorrow or use a new API key.")
                st.session_state.recipes  = []
            elif not result:
                st.error("No recipes found. Try different ingredient names.")
                st.session_state.recipes  = []
            else:
                st.session_state.recipes  = result
                st.session_state.analyzed = True

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5 — Recipe Results
    # ─────────────────────────────────────────────────────────────────────────
    if st.session_state.recipes:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">🎉 Step 5 — Your Recipes</p>', unsafe_allow_html=True)

        ingr_label = ", ".join(st.session_state.final_ingredients)
        st.markdown(
            f"<p style='color:rgba(255,255,255,0.4);font-size:0.88rem;margin-bottom:1rem'>"
            f"Found <b style='color:#ffd200'>{len(st.session_state.recipes)}</b> recipes "
            f"matching <b style='color:#ffd200'>{ingr_label}</b></p>",
            unsafe_allow_html=True,
        )

        for i, recipe in enumerate(st.session_state.recipes):
            title  = recipe.get("title", "Untitled")
            imgurl = recipe.get("image", "")
            used   = recipe.get("usedIngredientCount", 0)
            missed = recipe.get("missedIngredientCount", 0)
            likes  = recipe.get("likes", 0)

            used_names   = [u["name"] for u in recipe.get("usedIngredients",   [])]
            missed_names = [m["name"] for m in recipe.get("missedIngredients", [])]

            used_pills   = "".join(f'<span class="ingr-pill" style="font-size:.75rem">{n}</span>'   for n in used_names[:6])
            missed_pills = "".join(f'<span class="miss-pill">{n}</span>' for n in missed_names[:5])

            st.markdown(f"""
            <div class="recipe-card">
                <div class="recipe-title">#{i+1} &nbsp;{title}</div>
                <div class="recipe-meta">
                    ✅ Used: <b>{used}</b> &nbsp;|&nbsp;
                    🛒 Missing: <b>{missed}</b> &nbsp;|&nbsp;
                    ❤️ {likes} likes
                </div>
            </div>
            """, unsafe_allow_html=True)

            img_c, info_c = st.columns([1, 1.6])
            with img_c:
                if imgurl:
                    st.image(imgurl, use_container_width=True)
            with info_c:
                if used_names:
                    st.markdown("**✅ Ingredients you have:**")
                    st.markdown(used_pills, unsafe_allow_html=True)
                if missed_names:
                    st.markdown("**🛒 Still need:**")
                    st.markdown(missed_pills, unsafe_allow_html=True)
                if not missed_names:
                    st.markdown("🎉 **You have everything for this recipe!**")

            st.markdown("<hr>", unsafe_allow_html=True)

    elif st.session_state.analyzed:
        st.info("No recipes found. Try editing the ingredient names.")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center;color:rgba(255,255,255,0.2);font-size:0.78rem;padding:.4rem 0">
    AI Recipe Recommender · EfficientNet-B0 (best_model.pth) · Spoonacular API · Streamlit
</p>
""", unsafe_allow_html=True)