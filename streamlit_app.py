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
# IoU calculées
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
# Palette Cityscapes
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
st.markdown("""
This dashboard allows exploration of the dataset and comparison of segmentation models.

1. Use the **EDA tab** to explore semantic categories in the dataset.
2. Use the **COMPARE tab** to visually compare predictions and metrics.
""")

if "selected_label" not in st.session_state:
    st.session_state.selected_label = "All"

st.title("Cityscapes Segmentation Comparison")

tab1, tab2, tab3 = st.tabs(["EDA", "COMPARE", "POC"])

# =========================================================
# EDA TAB
# =========================================================
with tab1:

    st.header("Dataset exploration")

    st.info(
        "Select a semantic group to highlight its pixels across the segmentation masks. "
        "This helps visualize how major categories (vehicles, humans, nature...) "
        "are spatially distributed in the dataset."
    )

    # -------------------------
    # Fonction pour filtrer l'image par super-label
    # -------------------------
    def highlight_label(img, selected_label):
        if selected_label == "All":
            return img

        arr = np.array(img)
        target_labels = SUPER_LABELS.get(selected_label, [])
        colors = [np.array(rgb) for _, (name, rgb) in LABELS.items() if name in target_labels]

        mask = np.zeros(arr.shape[:2], dtype=bool)
        for c in colors:
            mask |= np.all(arr == c, axis=-1)

        white = np.ones_like(arr) * 255
        white[mask] = arr[mask]

        return Image.fromarray(white.astype(np.uint8))

    # -------------------------
    # Boutons pour super-labels
    # -------------------------
    st.subheader("Semantic groups")
    groups = ["All"] + list(SUPER_LABELS.keys())
    cols = st.columns(len(groups))
    for col, g in zip(cols, groups):
        if col.button(g, key=f"group_{g}"):
            st.session_state.selected_label = g
            st.rerun()
        if g != "All":
            color_boxes = "".join([
                f"<div style='width:18px;height:18px;background-color:rgb{rgb};display:inline-block;margin:2px;'></div>"
                for _, (name, rgb) in LABELS.items() if name in SUPER_LABELS[g]
            ])
            col.markdown(color_boxes, unsafe_allow_html=True)

    st.divider()

    # -------------------------
    # Exemples de masks
    # -------------------------
    st.subheader("Segmentation masks examples")
    mask_names = ["cologne.png", "hamburg.png", "tubingen.png"]
    mask_cols = st.columns(3)
    for col, name in zip(mask_cols, mask_names):
        img_path = os.path.join(EDA_DIR, name)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            filtered = highlight_label(img, st.session_state.selected_label)
            col.image(filtered, caption=f"Segmentation mask example from {name.split('.')[0]}", use_container_width=True)

    st.divider()

    # -------------------------
    # Distribution super-labels
    # -------------------------
    st.subheader("PIxel distribution (interactive)")

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

    fig_super = px.bar(
        df_super,
        x="super_label",
        y="pixels",
        title="Pixels distribution per labels",
        text="pixels",
        color="super_label",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_super.update_layout(
        xaxis_title="Super-label",
        yaxis_title="Nombre de pixels",
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )

    st.plotly_chart(fig_super, use_container_width=True)

    st.divider()

    # -------------------------
    # Distribution labels détaillés
    # -------------------------
    st.subheader("Detailed labels distribution (interactive)")

    label_counts = {
        'object': 157653,
        'void': 104867,
        'human': 45077,
        'nature': 40843,
        'construction': 24511,
        'flat': 24376,
        'vehicle': 81504,
        'rectification border': 9408,
        'out of roi': 6300,
        'sky': 6234
    }

 

    # Fonction super-label
    def get_super_label(label):
        if label in ['flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle', 'void']:
            return label
        elif label in ['rectification border', 'out of roi']:
            return 'void'
        else:
            return 'object'  # fallback

    df_labels = pd.DataFrame({
        "label": list(label_counts.keys()),
        "pixels": list(label_counts.values())
    })
    df_labels["super_label"] = df_labels["label"].apply(get_super_label)
    df_labels = df_labels.sort_values("pixels", ascending=False)

    fig_labels = px.bar(
        df_labels,
        x="label",
        y="pixels",
        color="super_label",
        title="Labels Distribution",
        text="pixels",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_labels.update_layout(
        xaxis_title="Label",
        yaxis_title="Nombre de pixels",
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        legend_title="Super-label"
    )

    st.plotly_chart(fig_labels, use_container_width=True)

 # -------------------------
    # static distributions
    # -------------------------

    st.subheader("Dataset distributions")

    dist_cols = st.columns(2)

    dist_imgs = [
        "distribution_nb_objet.png",
        "distribution_aires.png",
    ]

    captions = [
        "Distribution of number of objects per image",
        "Distribution of object areas"
    ]

    for col, img_name, cap in zip(dist_cols, dist_imgs, captions):

        img_path = os.path.join(EDA_DIR, img_name)

        if os.path.exists(img_path):

            col.image(
                Image.open(img_path),
                caption=cap,
                use_container_width=True
            )

# =========================================================
# MODELS TAB
# =========================================================
with tab2:

    st.header("Model comparison")

    headers = [
        "Image",
        "Masque réel",
        "Unet",
        "mit_b0",
        "mit_b0 vs Unet",
        "mit_b3",
        "mit_b3 vs Unet",
        "mit_b5",
        "mit_b5 vs Unet",
    ]

    cols = st.columns(len(headers))
    for col, h in zip(cols, headers):
        col.markdown(f"**{h}**")

    for i in range(5):
        cols = st.columns(len(headers))

        paths = [
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

        for j, p in enumerate(paths):
            img_path = os.path.join(IMG_DIR, p)
            if os.path.exists(img_path):
                img = Image.open(img_path)
                cols[j].image(img, use_container_width=True)

            # Ajout IoU
            if j == 2:
                cols[j].markdown(f"IoU: **{IOU[i]['unet']:.3f}**")
            if j == 3:
                cols[j].markdown(f"IoU: **{IOU[i]['mit_b0']:.3f}**")
            if j == 5:
                cols[j].markdown(f"IoU: **{IOU[i]['mit_b3']:.3f}**")
            if j == 7:
                cols[j].markdown(f"IoU: **{IOU[i]['mit_b5']:.3f}**")

            # Pixels gagnés
            if j in [4,6,8]:
                model_name = ["mit_b0","mit_b3","mit_b5"][ [4,6,8].index(j) ]
                pixels = PIXELS_GAINES[i][model_name]
                cols[j].markdown(
                    f"<div style='text-align:center'>Pixels gagnés: {pixels}</div>",
                    unsafe_allow_html=True
                )

# =========================================================
# 🔥 POC TAB
# =========================================================
with tab3:

    st.header("Live inference (U-Net vs SegFormer)")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
