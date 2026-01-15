import streamlit as st
import requests
import numpy as np
from PIL import Image
import io
import base64

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

FASTAPI_URL = "https://Jabb-projet8-segmentation-api.hf.space/predict"

st.title("Projet 8 - Segmentation d'images")

uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "jpeg", "png"])

def colorize_mask(mask):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        out[mask == c] = PALETTE[c]
    return out

if uploaded_file is not None:
    # Afficher image originale
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Image originale", use_column_width=True)
    
    # Envoyer à l'API FastAPI
    files = {"image": (uploaded_file.name, uploaded_file, uploaded_file.type)}
    response = requests.post(FASTAPI_URL, files=files)
    
    if response.status_code == 200:
        mask2d = np.array(response.json()["prediction"])
        colored_mask = colorize_mask(mask2d)
        st.image(colored_mask, caption="Masque coloré", use_column_width=True)
    else:
        st.error(f"Erreur API : {response.text}")
