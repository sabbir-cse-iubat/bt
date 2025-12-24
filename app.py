# app.py — FHD-HybridNet Brain Tumor MRI (Streamlit, GitHub-ready)
# ✅ sample_images/ from repo
# ✅ Downloads 3 weights (.h5) from Google Drive using gdown
# ✅ FHD-HybridNet ensemble
# ✅ YOUR EXACT Grad-CAM pipeline:
#    make_brain_mask_from_image + enhance_heatmap + overlay_gradcam_on_image_fixed
# ✅ Fixes Streamlit Cloud error:
#    "dense expects 1 input but got 2"
#    by rebuilding models in code + loading ONLY weights from .h5 (no deserialization)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import cv2
import gdown
import tensorflow as tf

# ----------------------------
# 0) BASIC SETUP
# ----------------------------
st.set_page_config(page_title="FHD-HybridNet Brain Tumor MRI", layout="wide")

IMG_SIZE = (224, 224)

# Must match your training generator class order (sorted folder names)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

SAMPLE_DIR = "sample_images"
MODEL_CACHE_DIR = "models_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# ----------------------------
# 1) GOOGLE DRIVE IDs (.h5)
# ----------------------------
# Paste your real Google Drive FILE IDs here (the part after /d/)
# Example:
# https://drive.google.com/file/d/1ABC...xyz/view -> ID=1ABC...xyz

H5_DENSENET_ID  = "1U7KwBmM7syLrU_F1aK0XThiMbAdHmfLC"
H5_MOBILENET_ID = "1c5A7PQf7WF1ZiJFlHM3s6oFusQRvHjR0"
H5_RESNET_ID    = "1LJEBxqHg_t9dQp1g6BykXGBIc1e7xZT0"

MODEL_FILES = {
    "DenseNet121": ("bt_model_a_best.h5", H5_DENSENET_ID),
    "MobileNetV1": ("bt_model_b_best.h5", H5_MOBILENET_ID),
    "ResNet50V2":  ("bt_model_c_best.h5", H5_RESNET_ID),
}

def gdrive_direct_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?id={file_id}"

def _is_probably_html(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(4096).lower()
        return (b"<html" in head) or (b"<!doctype html" in head) or (b"google drive" in head)
    except Exception:
        return False

def _validate_download(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File missing after download: {path}")
    size = os.path.getsize(path)
    if size < 50_000 or _is_probably_html(path):
        raise RuntimeError(
            "Downloaded file is not a valid .h5.\n\n"
            "Fix:\n"
            "1) Google Drive -> Share -> Anyone with the link (Viewer)\n"
            "2) Use correct FILE ID\n"
            f"Bad file: {path} ({size} bytes)"
        )

def _download_if_needed(file_id: str, filename: str) -> str:
    if not file_id or "PASTE_" in file_id:
        raise RuntimeError(
            "H5 model IDs are not set.\n"
            "Open app.py and replace H5_DENSENET_ID / H5_MOBILENET_ID / H5_RESNET_ID with your real Drive FILE IDs."
        )

    local_path = os.path.join(MODEL_CACHE_DIR, filename)

    if os.path.exists(local_path):
        try:
            _validate_download(local_path)
            return local_path
        except Exception:
            try:
                os.remove(local_path)
            except Exception:
                pass

    url = gdrive_direct_url(file_id)
    gdown.download(url, local_path, quiet=True, fuzzy=True)
    _validate_download(local_path)
    return local_path

# ----------------------------
# 2) BUILD MODELS (exact training head)
# ----------------------------
def _build_head(backbone):
    # matches your notebook:
    # Sequential([base, GAP, Dense(256), BN, Dropout(0.4), Dense(256), Dense(NUM_CLASSES, softmax)])
    return tf.keras.Sequential([
        backbone,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

def build_model_by_name(model_name: str):
    if model_name == "DenseNet121":
        base = tf.keras.applications.DenseNet121(include_top=False, weights=None, input_shape=(224, 224, 3))
        return _build_head(base)

    if model_name == "MobileNetV1":
        base = tf.keras.applications.MobileNet(include_top=False, weights=None, input_shape=(224, 224, 3))
        return _build_head(base)

    if model_name == "ResNet50V2":
        base = tf.keras.applications.ResNet50V2(include_top=False, weights=None, input_shape=(224, 224, 3))
        return _build_head(base)

    raise ValueError(f"Unknown model_name: {model_name}")

def load_model_weights_safe(model_name: str, h5_path: str):
    # rebuild model graph in code
    model = build_model_by_name(model_name)
    _ = model(tf.zeros((1, 224, 224, 3), dtype=tf.float32), training=False)

    # load weights only (prevents your dense input error)
    model.load_weights(h5_path)
    return model

@st.cache_resource(show_spinner=False)
def load_single_model(model_name: str):
    fname, file_id = MODEL_FILES[model_name]
    local_path = _download_if_needed(file_id, fname)
    return load_model_weights_safe(model_name, local_path)

@st.cache_resource(show_spinner=False)
def load_all_models():
    return {
        "DenseNet121": load_single_model("DenseNet121"),
        "MobileNetV1": load_single_model("MobileNetV1"),
        "ResNet50V2":  load_single_model("ResNet50V2"),
    }

# ----------------------------
# 3) IMAGE HELPERS
# ----------------------------
def load_image_from_file(file_or_path, img_size=IMG_SIZE):
    if isinstance(file_or_path, str):
        img = Image.open(file_or_path).convert("RGB")
    else:
        img = Image.open(file_or_path).convert("RGB")
    img_resized = img.resize(img_size)
    arr = np.asarray(img_resized).astype("float32") / 255.0
    batch = np.expand_dims(arr, axis=0)
    return img, batch

# ----------------------------
# 4) YOUR EXACT GRADCAM PIPELINE
# ----------------------------
def make_brain_mask_from_image(orig_pil):
    img = np.array(orig_pil.convert("L"))
    img = cv2.GaussianBlur(img, (5,5), 0)

    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if num_labels <= 1:
        mask = (th > 0).astype(np.float32)
        return mask

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest).astype(np.float32)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.erode(mask, k, iterations=1)

    mask = cv2.GaussianBlur(mask, (9,9), 0)
    mask = mask / (mask.max() + 1e-8)
    return mask.astype(np.float32)

def enhance_heatmap(heatmap, brain_mask=None, gamma=1.8, keep_percentile=85, keep_largest_blob=True):
    h = heatmap.astype(np.float32)
    h = np.maximum(h, 0)
    h = h / (h.max() + 1e-8)

    if brain_mask is not None:
        if brain_mask.shape != h.shape:
            brain_mask = cv2.resize(brain_mask, (h.shape[1], h.shape[0]))
        h = h * brain_mask
        h = h / (h.max() + 1e-8)

    h = np.power(h, gamma)

    thr = np.percentile(h, keep_percentile)
    h = np.where(h >= thr, h, 0.0)

    if keep_largest_blob:
        binmap = (h > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binmap, connectivity=8)
        if num_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            h = h * (labels == largest).astype(np.float32)

    h = h / (h.max() + 1e-8)
    return h

def overlay_gradcam_on_image_fixed(heatmap, orig_image, alpha=0.45):
    w, h = orig_image.size
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_resized = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    base = np.asarray(orig_image).astype("float32") / 255.0
    overlay = alpha * (heatmap_color.astype("float32") / 255.0) + (1 - alpha) * base
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay

def get_backbone(seq_model):
    if hasattr(seq_model, "layers") and len(seq_model.layers) >= 2:
        return seq_model.layers[0]
    return None

def pick_rank4_feature_layer(backbone, input_shape=(224,224,3)):
    x = tf.keras.Input(shape=input_shape)
    _ = backbone(x, training=False)

    candidates = []
    for i, layer in enumerate(backbone.layers):
        try:
            out = layer.output
            if out is None:
                continue
            shp = out.shape
            if shp is not None and len(shp) == 4:
                lname = layer.name.lower()
                bonus = 1 if ("conv" in lname or "block" in lname or "mixed" in lname or "relu" in lname) else 0
                candidates.append((i, bonus, layer.name))
        except Exception:
            pass

    if not candidates:
        raise ValueError("No rank-4 feature layer found in backbone.")
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]

def gradcam_hooked(seq_model, img_batch, class_index=None):
    backbone = get_backbone(seq_model)
    if backbone is None:
        raise ValueError("Model does not look like Sequential([backbone, head...]).")

    feat_layer_name = pick_rank4_feature_layer(backbone, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    target_layer = backbone.get_layer(feat_layer_name)

    conv_out = {"val": None}
    original_call = target_layer.call

    def wrapped_call(*args, **kwargs):
        out = original_call(*args, **kwargs)
        conv_out["val"] = out
        return out

    target_layer.call = wrapped_call

    x = tf.convert_to_tensor(img_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        preds = seq_model(x, training=False)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        loss = preds[:, class_index]
        tape.watch(conv_out["val"])

    grads = tape.gradient(loss, conv_out["val"])
    target_layer.call = original_call

    if conv_out["val"] is None or grads is None:
        raise RuntimeError("Failed to capture feature map / gradients for Grad-CAM.")

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_map = conv_out["val"][0]

    heatmap = tf.reduce_sum(conv_map * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), preds.numpy()[0]


# ----------------------------
# 5) FHD ENSEMBLE (same logic)
# ----------------------------
def fuzzy_hellinger_distance(p1, p2):
    return 0.5 * np.sum((np.sqrt(p1) - np.sqrt(p2)) ** 2)

def predict_probs(model, img_batch):
    x = tf.convert_to_tensor(img_batch, dtype=tf.float32)
    return model(x, training=False).numpy()[0]

def ensemble_predict_fhd_single(models_dict, img_batch):
    keys = ["DenseNet121", "MobileNetV1", "ResNet50V2"]
    preds_list = [predict_probs(models_dict[k], img_batch) for k in keys]

    avg_fhd = []
    for i in range(3):
        d = []
        for j in range(3):
            if i == j:
                continue
            d.append(fuzzy_hellinger_distance(preds_list[i], preds_list[j]))
        avg_fhd.append(float(np.mean(d)))

    best_idx = int(np.argmin(avg_fhd))
    chosen_key = keys[best_idx]
    chosen_probs = preds_list[best_idx]
    pred_idx = int(np.argmax(chosen_probs))
    return pred_idx, chosen_probs, chosen_key

# ----------------------------
# 6) SIDEBAR UI
# ----------------------------
st.sidebar.title("Controls")

model_name = st.sidebar.selectbox(
    "Select model",
    ["DenseNet121", "MobileNetV1", "ResNet50V2", "FHD-HybridNet"],
    index=3
)

source = st.sidebar.radio("Choose image source", ["Upload MRI", "Sample gallery"], index=1)

chosen_file = None
if source == "Upload MRI":
    uploaded = st.sidebar.file_uploader("Upload an MRI image", type=["png","jpg","jpeg","webp"])
    if uploaded is not None:
        chosen_file = uploaded
else:
    if os.path.isdir(SAMPLE_DIR):
        gallery_files = sorted([f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".png",".jpg",".jpeg",".webp"))])
    else:
        gallery_files = []

    if gallery_files:
        sample_name = st.sidebar.selectbox("Pick a sample image", gallery_files)
        chosen_file = os.path.join(SAMPLE_DIR, sample_name)
    else:
        st.sidebar.warning("No images found in sample_images/. Commit images into repo.")

run_button = st.sidebar.button("▶ Run prediction")

# ----------------------------
# 7) MAIN HEADER
# ----------------------------
st.title("FHD-HybridNet Brain Tumor MRI Classification")
st.markdown(
    """
This app uses three CNN backbones (**DenseNet121, MobileNetV1, ResNet50V2**)  
and a fuzzy-logic-based ensemble (**Fuzzy Hellinger Distance**)  
to classify Brain Tumor MRI scans into **four classes**.
"""
)

if not run_button:
    st.stop()

if chosen_file is None:
    st.error("Please upload or select an image first.")
    st.stop()

orig_img, batch = load_image_from_file(chosen_file, IMG_SIZE)

# ----------------------------
# 8) INFERENCE + GRADCAM
# ----------------------------
try:
    with st.spinner("Running prediction…"):
        if model_name == "FHD-HybridNet":
            models_dict = load_all_models()
            pred_idx, probs, chosen_key = ensemble_predict_fhd_single(models_dict, batch)
            cam_model = models_dict[chosen_key]
            cam_title = f"FHD-HybridNet (chosen: {chosen_key})"
        else:
            cam_model = load_single_model(model_name)
            probs = predict_probs(cam_model, batch)
            pred_idx = int(np.argmax(probs))
            cam_title = model_name

    pred_class = CLASS_NAMES[pred_idx]

    with st.spinner("Computing Grad-CAM…"):
        heatmap, _ = gradcam_hooked(cam_model, batch, class_index=pred_idx)
        brain_mask = make_brain_mask_from_image(orig_img)
        heatmap2 = enhance_heatmap(heatmap, brain_mask=brain_mask, gamma=1.8, keep_percentile=85, keep_largest_blob=True)
        overlay = overlay_gradcam_on_image_fixed(heatmap2, orig_img, alpha=0.45)

except Exception as e:
    st.error("❌ Failed.")
    st.code(str(e))
    st.stop()

# ----------------------------
# 9) OUTPUT
# ----------------------------
st.markdown("---")
st.subheader("Prediction Output")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(orig_img)
axes[0].set_title(f"Original\nPredicted: {pred_class}")
axes[0].axis("off")

axes[1].imshow(overlay)
axes[1].set_title(f"Grad-CAM\n{cam_title}")
axes[1].axis("off")

axes[2].barh(CLASS_NAMES, probs)
axes[2].set_xlim(0, 1)
axes[2].set_xlabel("Probability")
axes[2].set_title("Probabilities")
for i, cls in enumerate(CLASS_NAMES):
    axes[2].text(float(probs[i]) + 0.01, i, f"{float(probs[i]):.3f}", va="center")

plt.tight_layout()
st.pyplot(fig)

st.markdown(f"#### Final Prediction : *{pred_class}*")

buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
buf.seek(0)

st.download_button(
    "💾 Save result image",
    data=buf,
    file_name=f"result_{pred_class}.png",
    mime="image/png"
)
