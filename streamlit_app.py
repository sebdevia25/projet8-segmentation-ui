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

# =========================
# IoU
# =========================
IOU = {
    0: {"unet":0.5769, "mit_b0":0.6998, "mit_b3":0.7953, "mit_b5":0.6916},
    1: {"unet":0.7021, "mit_b0":0.7306, "mit_b3":0.7936, "mit_b5":0.7970},
    2: {"unet":0.5614, "mit_b0":0.6189, "mit_b3":0.6766, "mit_b5":0.6659},
    3: {"unet":0.7578, "mit_b0":0.7657, "mit_b3":0.7974, "mit_b5":0.7969},
    4: {"unet":0.6105, "mit_b0":0.7121, "mit_b3":0.7515, "mit_b5":0.7680},
}

# =========================
# Pixels gagnés
# =========================
PIXELS_GAINES = {
    0: {"mit_b0":26337, "mit_b3":29910, "mit_b5":31704},
    1: {"mit_b0":12694, "mit_b3":18445, "mit_b5":19193},
    2: {"mit_b0":23309, "mit_b3":23355, "mit_b5":26196},
    3: {"mit_b0":11564, "mit_b3":14140, "mit_b5":14372},
    4: {"mit_b0":41540, "mit_b3":44174, "mit_b5":47677},
}

# =========================
# Cityscapes labels
# =========================
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

# =========================
# Utils
# =========================
def colorize_mask(mask2d):
    mask2d = np.array(mask2d)
    h, w = mask2d.shape

    rng = np.random.RandomState(0)
    palette = rng.randint(0, 255, (20, 3), dtype=np.uint8)

    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(20):
        out[mask2d == c] = palette[c]
    return out


def encode_image(img):
    pil = Image.fromarray(img)
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# =========================
# Streamlit config
# =========================
st.set_page_config(layout="wide")
st.title("Cityscapes Segmentation Comparison")

tab1, tab2, tab3 = st.tabs(["EDA", "MODELS", "POC"])

# =========================================================
# EDA TAB
# =========================================================
with tab1:

    st.header("Dataset exploration")

    if "selected_label" not in st.session_state:
        st.session_state.selected_label = "All"

    def highlight_label(img, selected_label):
        if selected_label == "All":
            return img

        arr = np.array(img)
        target_labels = SUPER_LABELS.get(selected_label, [])
        colors = [
            np.array(rgb)
            for _, (name, rgb) in LABELS.items()
            if name in target_labels
        ]

        mask = np.zeros(arr.shape[:2], dtype=bool)
        for c in colors:
            mask |= np.all(arr == c, axis=-1)

        white = np.ones_like(arr) * 255
        white[mask] = arr[mask]
        return Image.fromarray(white.astype(np.uint8))

    st.subheader("Semantic groups")

    groups = ["All"] + list(SUPER_LABELS.keys())
    cols = st.columns(len(groups))

    for col, g in zip(cols, groups):
        if col.button(g):
            st.session_state.selected_label = g
            st.rerun()

    st.divider()

    mask_names = ["cologne.png", "hamburg.png", "tubingen.png"]
    cols = st.columns(3)

    for col, name in zip(cols, mask_names):
        path = os.path.join(EDA_DIR, name)
        if os.path.exists(path):
            img = Image.open(path).convert("RGB")
            img = highlight_label(img, st.session_state.selected_label)
            col.image(img, use_container_width=True)

    st.divider()

    real_counts = {
        'flat': 338835.19,
        'human': 4210.86,
        'vehicle': 15364.31,
        'construction': 218146.78,
        'object': 1640.91,
        'nature': 62174.63,
        'void': 5683.68
    }

    df_super = pd.DataFrame({
        "super_label": list(real_counts.keys()),
        "pixels": list(real_counts.values())
    })

    fig = px.bar(df_super, x="super_label", y="pixels", color="super_label")
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# MODELS TAB
# =========================================================
with tab2:

    st.header("Model comparison")

    headers = [
        "Image","Mask","Unet","mit_b0","b0 vs Unet",
        "mit_b3","b3 vs Unet","mit_b5","b5 vs Unet"
    ]

    cols = st.columns(len(headers))
    for c, h in zip(cols, headers):
        c.markdown(f"**{h}**")

    for i in range(5):

        cols = st.columns(len(headers))

        files = [
            f"image_{i}.png",
            f"masque_{i}.png",
            f"pred_unet_{i}.png",
            f"pred_mit_b0_{i}.png",
            f"comp_mit_b0_{i}.png",
            f"pred_mit_b3_{i}.png",
            f"comp_mit_b3_{i}.png",
            f"pred_mit_b5_{i}.png",
            f"comp_mit_b5_{i}.png",
        ]

        for j, f in enumerate(files):

            path = os.path.join(IMG_DIR, f)

            if os.path.exists(path):
                cols[j].image(Image.open(path), use_container_width=True)

            if j == 2:
                cols[j].markdown(f"IoU {IOU[i]['unet']:.3f}")
            if j == 3:
                cols[j].markdown(f"IoU {IOU[i]['mit_b0']:.3f}")
            if j == 5:
                cols[j].markdown(f"IoU {IOU[i]['mit_b3']:.3f}")
            if j == 7:
                cols[j].markdown(f"IoU {IOU[i]['mit_b5']:.3f}")

            if j in [4,6,8]:
                model = ["mit_b0","mit_b3","mit_b5"][[4,6,8].index(j)]
                cols[j].markdown(f"Pixels: {PIXELS_GAINES[i][model]}")

# =========================================================
# POC TAB
# =========================================================
with tab3:

    st.header("Live inference (U-Net vs SegFormer)")

    file = st.file_uploader("Upload image", type=["png","jpg","jpeg"])

    if file:

        img_bytes = file.read()

        st.image(img_bytes, caption="Original", use_container_width=True)

        files = {
            "image": (file.name, img_bytes, file.type)
        }

        col1, col2 = st.columns(2)

        # ---------------- UNET ----------------
        with col1:
            st.subheader("U-Net")

            r = requests.post(FASTAPI_UNET_URL, files=files)

            if r.status_code == 200:
                data = r.json()
                mask = base64.b64decode(data["mask_base64"])
                st.image(mask)
            else:
                st.error(r.text)

        # ---------------- SEGFORMER ----------------
        with col2:
            st.subheader("SegFormer B3")

            r = requests.post(FASTAPI_SEGFORMER_URL, files=files)

            if r.status_code == 200:
                data = r.json()
                mask = base64.b64decode(data["mask_base64"])
                st.image(mask)
            else:
                st.error(r.text)
