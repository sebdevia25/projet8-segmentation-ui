import streamlit as st
import requests
import numpy as np
from PIL import Image

# =====================
# CONFIG
# =====================
FASTAPI_URL = "https://Jabb-projet8-segmentation-api.hf.space/predict"

NUM_CLASSES = 8
PALETTE = np.array([
    [0, 0, 0],         # void
    [128, 64, 128],    # flat
    [220, 20, 60],     # human
    [0, 0, 142],       # vehicle
    [70, 70, 70],      # construction
    [153, 153, 153],   # object
    [107, 142, 35],    # nature
    [70, 130, 180],    # sky
], dtype=np.uint8)

SAMPLES = {
    "Ville 1": ("samples/image_1.jpg", "samples/mask_1.png"),
    "Ville 2": ("samples/image_2.jpg", "samples/mask_2.png"),
    "Ville 3": ("samples/image_3.jpg", "samples/mask_3.png"),
    "Ville 4": ("samples/image_4.jpg", "samples/mask_4.png"),
    "Ville 5": ("samples/image_5.jpg", "samples/mask_5.png"),
}

# =====================
# UTILS
# =====================
def colorize_mask(mask):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        out[mask == c] = PALETTE[c]
    return out

def overlay_image(image, mask, alpha=0.5):
    """Superposition image + masque"""
    return Image.blend(image, mask, alpha)

# =====================
# UI
# =====================
st.set_page_config(page_title="Segmentation urbaine", layout="wide")
st.title("Segmentation sémantique urbaine")

mode = st.radio(
    "Choisir une image",
    ["Uploader une image", "Utiliser une image de démonstration"]
)

alpha = st.slider("Transparence du masque", 0.0, 1.0, 0.5, 0.05)

# =====================
# IMAGE SOURCE
# =====================
if mode == "Uploader une image":
    uploaded_file = st.file_uploader("Image (jpg / png)", type=["jpg", "jpeg", "png"])
    gt_mask = None

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

else:
    sample_name = st.selectbox("Image de démonstration", list(SAMPLES.keys()))
    img_path, mask_path = SAMPLES[sample_name]

    image = Image.open(img_path).convert("RGB")
    gt_mask = Image.open(mask_path).convert("RGB")

# =====================
# PREDICTION
# =====================
if image:
    col1, col2, col3 = st.columns(3)

    col1.image(image, caption="Image originale", use_column_width=True)

    # Appel API
    img_bytes = image.copy()
    buffer = img_bytes.tobytes()

    response = requests.post(
        FASTAPI_URL,
        files={"image": ("image.png", image.tobytes(), "image/png")}
    )

    if response.status_code != 200:
        st.error(f"Erreur API : {response.text}")
        st.stop()

    mask2d = np.array(response.json()["prediction"])
    colored_mask = colorize_mask(mask2d)

    mask_img = Image.fromarray(colored_mask)
    mask_img = mask_img.resize(image.size, Image.NEAREST)

    col2.image(mask_img, caption="Masque prédit", use_column_width=True)

    overlay = overlay_image(image, mask_img, alpha)
    col3.image(overlay, caption="Superposition", use_column_width=True)

    # =====================
    # GROUND TRUTH
    # =====================
    if gt_mask:
        st.subheader("Masque de référence (entraînement)")
        gt_mask = gt_mask.resize(image.size, Image.NEAREST)
        st.image(gt_mask, use_column_width=True)
