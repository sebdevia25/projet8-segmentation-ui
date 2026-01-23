import streamlit as st
import requests
import numpy as np
from PIL import Image
import io
import os

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

BASE_DIR = os.path.dirname(__file__)

SAMPLES = {
    "Ville 1": ("samples/image_1.png", "samples/mask_1.png"),
    "Ville 2": ("samples/image_2.png", "samples/mask_2.png"),
    "Ville 3": ("samples/image_3.png", "samples/mask_3.png"),
    "Ville 4": ("samples/image_4.png", "samples/mask_4.png"),
    "Ville 5": ("samples/image_5.png", "samples/mask_5.png"),
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
    return Image.blend(image, mask, alpha)

def send_image_to_api(pil_img):
    """Encode correctement une image PIL et l'envoie à l'API"""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    response = requests.post(
        FASTAPI_URL,
        files={"image": ("image.png", buf, "image/png")},
        timeout=30
    )
    return response

def compute_iou(mask_pred, mask_gt, num_classes=8):
    """Calcule l'IoU moyen global entre le masque prédit et le masque ground truth."""
    ious = []
    for c in range(num_classes):
        pred_c = (mask_pred == c)
        gt_c = (mask_gt == c)
        intersection = np.logical_and(pred_c, gt_c).sum()
        union = np.logical_or(pred_c, gt_c).sum()
        if union == 0:
            continue  # ignorer les classes absentes
        ious.append(intersection / union)
    if len(ious) == 0:
        return 0.0
    return np.mean(ious)

# =====================
# UI
# =====================
st.set_page_config(page_title="Segmentation sémantique", layout="wide")
st.title("Projet 8 — Segmentation sémantique urbaine")

mode = st.radio(
    "Choisir une image",
    ["Uploader une image", "Utiliser une image de démonstration"]
)

alpha = st.slider("Transparence du masque", 0.0, 1.0, 0.5, 0.05)

image = None
gt_mask = None

# =====================
# IMAGE SOURCE
# =====================
if mode == "Uploader une image":
    uploaded_file = st.file_uploader("Image (JPG / PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

elif mode == "Utiliser une image de démonstration":
    sample_name = st.selectbox("Image de démonstration", list(SAMPLES.keys()))
    img_path, mask_path = SAMPLES[sample_name]

    img_path = os.path.join(BASE_DIR, img_path)
    mask_path = os.path.join(BASE_DIR, mask_path)

    image = Image.open(img_path).convert("RGB")
    gt_mask = Image.open(mask_path).convert("RGB")

# =====================
# PREDICTION
# =====================
if image is not None:
    col1, col2, col3 = st.columns(3)

    col1.image(image, caption="Image originale", use_column_width=True)

    with st.spinner("Segmentation en cours..."):
        response = send_image_to_api(image)

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
    if gt_mask is not None:
        #st.subheader("Masque de référence (entraînement)")

        # Redimensionner exactement comme le masque prédit
        gt_mask_resized = gt_mask.resize(image.size, Image.NEAREST)
         # Convertir en array pour calculer IoU
        mask_pred_array = np.array(mask_img.resize(image.size, Image.NEAREST))
        mask_gt_array = np.array(gt_mask_resized)
        iou_score = compute_iou(mask_pred_array[:,:,0], mask_gt_array[:,:,0], NUM_CLASSES)
        # Afficher juste en dessous du masque prédit
        col_gt1, col_gt2, col_gt3 = st.columns(3)
      
        col_gt2.image(gt_mask.resize(image.size, Image.NEAREST), caption="Masque de référence", use_column_width=True)
        col_gt2.metric(label="IoU global", value=f"{iou_score:.2f}")