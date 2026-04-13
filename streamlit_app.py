import streamlit as st
from PIL import Image
import numpy as np
import os
import plotly.express as px
import pandas as pd
import requests
import io
import base64

IMG_DIR = "images"
EDA_DIR = "images/EDA"

# =========================
# API URLs
# =========================
FASTAPI_UNET_URL = "https://Jabb-projet8-segmentation-api.hf.space/predict"
FASTAPI_SEGFORMER_URL = "https://Jabb-projet8-segmentation-api.hf.space/predict-segformer"

# -------------------------
# IoU
# -------------------------
IOU = {
    0: {"unet":0.5769, "mit_b0":0.6998, "mit_b3":0.7953, "mit_b5":0.6916},
    1: {"unet":0.7021, "mit_b0":0.7306, "mit_b3":0.7936, "mit_b5":0.7970},
    2: {"unet":0.5614, "mit_b0":0.6189, "mit_b3":0.6766, "mit_b5":0.6659},
    3: {"unet":0.7578, "mit_b0":0.7657, "mit_b3":0.7974, "mit_b5":0.7969},
    4: {"unet":0.6105, "mit_b0":0.7121, "mit_b3":0.7515, "mit_b5":0.7680},
}

# -------------------------
# Pixels gagnés
# -------------------------
PIXELS_GAINES = {
    0: {"mit_b0":26337, "mit_b3":29910, "mit_b5":31704},
    1: {"mit_b0":12694, "mit_b3":18445, "mit_b5":19193},
    2: {"mit_b0":23309, "mit_b3":23355, "mit_b5":26196},
    3: {"mit_b0":11564, "mit_b3":14140, "mit_b5":14372},
    4: {"mit_b0":41540, "mit_b3":44174, "mit_b5":47677},
}

# -------------------------
# Labels Cityscapes
# -------------------------
LABELS = {
    0: ("road",(128,64,128)),
    1: ("sidewalk",(244,35,232)),
    2: ("building",(70,70,70)),
    3: ("wall",(102,102,156)),
    4: ("fence",(190,153,153)),
    5: ("pole",(153,153,153)),
    6: ("traffic light",(250,170,30)),
    7: ("traffic sign",(220,220,0)),
    8: ("vegetation",(107,142,35)),
    9: ("terrain",(152,251,152)),
    10: ("sky",(70,130,180)),
    11: ("person",(220,20,60)),
    12: ("rider",(255,0,0)),
    13: ("car",(0,0,142)),
    14: ("truck",(0,0,70)),
    15: ("bus",(0,60,100)),
    16: ("train",(0,80,100)),
    17: ("motorcycle",(0,0,230)),
    18: ("bicycle",(119,11,32)),
}

SUPER_LABELS = {
    "flat": ["road", "sidewalk", "terrain"],
    "construction": ["building", "wall", "fence"],
    "object": ["pole", "traffic light", "traffic sign"],
    "nature": ["vegetation"],
    "sky": ["sky"],
    "human": ["person", "rider"],
    "vehicle": ["car", "truck", "bus", "train", "motorcycle", "bicycle"],
    "void": [],
}

st.set_page_config(layout="wide")

st.title("Cityscapes Segmentation Comparison")

tab1, tab2, tab3 = st.tabs(["EDA", "MODELS", "POC"])

# =========================================================
# EDA TAB (inchangé)
# =========================================================
with tab1:
    st.header("Dataset exploration")
    st.info("Explore dataset distributions and masks.")
    st.write("... (ton code EDA inchangé) ...")

# =========================================================
# MODELS TAB (inchangé)
# =========================================================
with tab2:
    st.header("Model comparison")
    st.write("... (ton code MODELS inchangé) ...")

# =========================================================
# 🔥 POC TAB
# =========================================================
with tab3:

    st.header("Live inference (U-Net vs SegFormer)")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file:

        image_bytes = uploaded_file.read()

        st.image(image_bytes, caption="Original image", use_container_width=True)

        files = {
            "image": (uploaded_file.name, image_bytes, uploaded_file.type)
        }

        col1, col2 = st.columns(2)

        # =========================
        # U-NET
        # =========================
        with col1:
            st.subheader("U-Net")

            try:
                r = requests.post(FASTAPI_UNET_URL, files=files)

                if r.status_code == 200:
                    data = r.json()

                    mask = base64.b64decode(data["mask_base64"])
                    st.image(mask, caption="U-Net prediction")
                else:
                    st.error(r.text)

            except Exception as e:
                st.error(f"U-Net error: {e}")

        # =========================
        # SEGFORMER
        # =========================
        with col2:
            st.subheader("SegFormer B3")

            try:
                r = requests.post(FASTAPI_SEGFORMER_URL, files=files)

                if r.status_code == 200:
                    data = r.json()

                    mask = base64.b64decode(data["mask_base64"])
                    st.image(mask, caption="SegFormer prediction")
                else:
                    st.error(r.text)

            except Exception as e:
                st.error(f"SegFormer error: {e}")
