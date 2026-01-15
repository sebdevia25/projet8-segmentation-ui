import streamlit as st
import requests
import numpy as np
from PIL import Image
import io

NUM_CLASSES = 8
PALETTE = [
    [0, 0, 0],
    [128, 64, 128],
    [220, 20, 60],
    [0, 0, 142],
    [70, 70, 70],
    [153, 153, 153],
    [107, 142, 35],
    [70, 130, 180],
]

# URL de ton API FastAPI HF
FASTAPI_URL = "https://Jabb-projet8-segmentation-api.hf.space/predict"

st.title("Projet 8 - Segmentation d'images")

uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "jpeg", "png"])

def colorize_mask(mask):
    """Convertit un mask 2D en image RGB colorée."""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        out[mask == c] = PALETTE[c]
    return out

if uploaded_file is not None:
    # Ouvrir image originale
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Image originale", use_column_width=True)

    # Préparer le fichier pour requests
    uploaded_file.seek(0)  # remettre le curseur au début
    files = {"image": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}

    # Appel à l'API FastAPI
    response = requests.post(FASTAPI_URL, files=files)

    if response.status_code == 200:
        mask2d = np.array(response.json()["prediction"])
        colored_mask = colorize_mask(mask2d)

        # Redimensionner le mask à la taille originale de l'image
        mask_img = Image.fromarray(colored_mask)
        mask_img = mask_img.resize(img.size, resample=Image.NEAREST)

        st.image(mask_img, caption="Masque coloré", use_column_width=True)
    else:
        st.error(f"Erreur API : {response.text}")
