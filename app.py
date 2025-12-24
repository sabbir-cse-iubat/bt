# app.py — FHD-HybridNet Brain Tumor MRI (Streamlit, GitHub-ready)
# - Uses sample_images/ from repo (you will commit images manually)
# - Downloads 3 models from Google Drive (gdown)
# - Runs single model or FHD-HybridNet ensemble
# - Grad-CAM (hook-based for Sequential([backbone, head...])) + brain mask + heatmap enhancement
#
# IMPORTANT:
# - Streamlit Cloud should use Python 3.10/3.11 for TensorFlow stability.
# - Keep TF_USE_LEGACY_KERAS = 0 (Keras 3)
#
# If you get: "dense expects 1 input but got 2"
# -> Your saved .keras is not loading cleanly in this environment.
# -> Fix without retrain: load in Colab then export to .h5, upload to Drive, update IDs.

import os
os.environ["TF_USE_LEGACY_KERAS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import time
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import cv2
import gdown
import tensorflow as tf
import keras  # Keras 3 (works with TF 2.16+)

# ----------------------------
# 0) BASIC SETUP
# ----------------------------
st.set_page_config(page_title="FHD-HybridNet Brain Tumor MRI", layout="wide")

IMG_SIZE = (224, 224)

# ⚠️ Must match training class order.
# If your training generator had different order, update it accordingly.
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

SAMPLE_DIR = "sample_images"
MODEL_CACHE_DIR = "models_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# ----------------------------
# 1) GOOGLE DRIVE LINKS (IDs)
# ----------------------------
# Your links:
# model a : https://drive.google.com/file/d/1IVbNJA_TKFsT9ZftiWyihxp6CYSoJxwd/view
# model b : https://drive.google.com/file/d/1MOXJSc3GuHoq4T7ZIPKNqyDF3r33_pml/view
# model c : https://drive.google.com/file/d/1zgfjc0JTIe1Xg24rcWTT7zym9FL2MObF/view

DENSENET_ID  = "1IVbNJA_TKFsT9ZftiWyihxp6CYSoJxwd"
MOBILENET_ID = "1MOXJSc3GuHoq4T7ZIPKNqyDF3r33_pml"
RESNET_ID    = "1zgfjc0JTIe1Xg24rcWTT7zym9FL2MObF"

MODEL_FILES = {
    "DenseNet121": ("bt_model_a.keras", DENSENET_ID),
    "MobileNetV1": ("bt_model_b.keras", MOBILENET_ID),
    "ResNet50V2":  ("bt_model_c.keras", RESNET_ID),
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
    if size < 50_000:
        if _is_probably_html(path):
            raise RuntimeError(
                "Downloaded file looks like an HTML page (Google Drive permission/confirm page), not a model.\n\n"
                "Fix:\n"
                "1) In Google Drive -> Share -> Anyone with the link (Viewer)\n"
                "2) Use the correct file ID\n"
                f"\nBad file: {path} ({size} bytes)"
            )
        raise RuntimeError(
            f"Downloaded file too small to be a valid model archive: {path} ({size} bytes).\n"
            "Fix: re-upload model, ensure link is public."
        )

    if _is_probably_html(path):
        raise RuntimeError(
            "Downloaded file looks like HTML (Drive page), not a real model.\n"
            "Fix: Make the file public: Anyone with the link (Viewer)."
        )

def _download_if_needed(file_id: str, filename: str) -> str:
    local_path = os.path.join(MODEL_CACHE_DIR, filename)

    # If exists, validate. If invalid remove and redownload.
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

    # Download with fuzzy=True (handles some GDrive patterns)
    gdown.download(url, local_path, quiet=True, fuzzy=True)
    _validate_download(local_path)
    return local_path

def _try_load_model(local_path: str):
    """
    Robust-ish loader:
    - Try Keras 3 native loader first
    - Fallback tf.keras loader
    """
    # 1) Keras 3 loader
    try:
        return keras.saving.load_model(local_path, compile=False)
    except Exception as e1:
        # 2) tf.keras loader
        try:
            return tf.keras.models.load_model(local_path, compile=False)
        except Exception as e2:
            raise RuntimeError(
                "Model load failed.\n\n"
                f"File: {local_path}\n\n"
                f"Keras loader error: {e1}\n\n"
                f"TF loader error: {e2}\n\n"
                "If you see an error like:\n"
                "  'dense expects 1 input but got 2'\n"
                "then the model archive is not compatible in this environment.\n\n"
                "✅ Fix WITHOUT retraining:\n"
                "1) Open Colab\n"
                "2) Load model there\n"
                "3) Re-save/export to .h5\n"
                "4) Upload .h5 to Drive and update IDs in app.py\n"
            )

@st.cache_resource(show_spinner=False)
def load_single_model(model_name: str):
    fname, file_id = MODEL_FILES[model_name]
    local_path = _download_if_needed(file_id, fname)
    model = _try_load_model(local_path)
    return model

@st.cache_resource(show_spinner=False)
def load_all_models():
    return {
        "DenseNet121": load_single_model("DenseNet121"),
        "MobileNetV1": load_single_model("MobileNetV1"),
        "ResNet50V2":  load_single_model("ResNet50V2"),
    }

# ----------------------------
# 2) IMAGE HELPERS
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
# 3) GRAD-CAM HELPERS (Your approach)
# ----------------------------
def make_brain_mask_from_image(orig_pil):
    img = np.array(orig_pil.convert("L"))
    img = cv2.GaussianBlur(img, (5, 5), 0)

    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if num_labels <= 1:
        return (th > 0).astype(np.float32)

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest).astype(np.float32)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.erode(mask, k, iterations=1)

    mask = cv2.GaussianBlur(mask, (9, 9), 0)
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
    # expects Sequential([backbone, head...])
    if hasattr(seq_model, "layers") and len(seq_model.layers) >= 2:
        return seq_model.layers[0]
    return None

def pick_rank4_feature_layer(backbone, input_shape=(224, 224, 3)):
    # Build backbone graph once
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
        raise ValueError("No rank-4 feature layer found in backbone (no HxWxC feature map).")
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]

def gradcam_hooked(seq_model, img_batch, class_index=None):
    backbone = get_backbone(seq_model)
    if backbone is None:
        raise ValueError("Model is not Sequential([backbone, head...]). Cannot run this Grad-CAM method.")

    feat_layer_name = pick_rank4_feature_layer(backbone, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    target_layer = backbone.get_layer(feat_layer_name)

    conv_out = {"val": None}
    original_call = target_layer.call

    def wrapped_call(*args, **kwargs):
        out = original_call(*args, **kwargs)
        conv_out["val"] = out
        return out

    # Temporarily wrap layer call to capture activations
    target_layer.call = wrapped_call

    x = tf.convert_to_tensor(img_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        preds = seq_model(x, training=False)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        loss = preds[:, class_index]

        # Ensure tape watches conv output
        if conv_out["val"] is None:
            # forward might not have hit layer (rare)
            _ = seq_model(x, training=False)
        tape.watch(conv_out["val"])

    grads = tape.gradient(loss, conv_out["val"])

    # Restore original call
    target_layer.call = original_call

    if conv_out["val"] is None or grads is None:
        raise RuntimeError("Failed to capture feature map/gradients for Grad-CAM.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)
    conv_map = conv_out["val"][0]                         # (H,W,C)

    heatmap = tf.reduce_sum(conv_map * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), feat_layer_name

# ----------------------------
# 4) FHD ENSEMBLE HELPERS
# ----------------------------
def fuzzy_hellinger_distance(p1, p2):
    return 0.5 * np.sum((np.sqrt(p1) - np.sqrt(p2)) ** 2)

def ensemble_predict_fhd_single(models_dict, img_batch):
    keys = ["DenseNet121", "MobileNetV1", "ResNet50V2"]
    preds_list = [models_dict[k].predict(img_batch, verbose=0)[0] for k in keys]

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
    return chosen_probs, pred_idx, chosen_key

def run_fhd_ensemble(img_batch):
    models_dict = load_all_models()
    probs, pred_idx, chosen_key = ensemble_predict_fhd_single(models_dict, img_batch)
    grad_model = models_dict[chosen_key]
    return probs, pred_idx, chosen_key, grad_model

# ----------------------------
# 5) SIDEBAR UI
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
    uploaded = st.sidebar.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is not None:
        chosen_file = uploaded
else:
    if os.path.isdir(SAMPLE_DIR):
        gallery_files = sorted([f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))])
    else:
        gallery_files = []

    if gallery_files:
        sample_name = st.sidebar.selectbox("Pick a sample image", gallery_files)
        chosen_file = os.path.join(SAMPLE_DIR, sample_name)
    else:
        st.sidebar.warning("No images found in sample_images/. Commit some images into repo.")

run_button = st.sidebar.button("▶ Run prediction")

# ----------------------------
# 6) MAIN HEADER
# ----------------------------
st.title("FHD-HybridNet Brain Tumor MRI Classification")
st.markdown(
    """
This app uses three CNN backbones (**DenseNet121, MobileNetV1, ResNet50V2**)  
and a fuzzy-logic-based ensemble (**Fuzzy Hellinger Distance**)  
to classify Brain Tumor MRI scans into **four classes**.
"""
)

col_info, _ = st.columns([1, 1])
with col_info:
    st.info("👈 Select a model & image, then click **Run prediction**.")
    st.write(f"**Model:** {model_name}")
    st.write(f"**Source:** {source}")

if not run_button:
    st.stop()

if chosen_file is None:
    st.error("Please upload or select an image first.")
    st.stop()

# ----------------------------
# 7) RUN INFERENCE
# ----------------------------
orig_img, batch = load_image_from_file(chosen_file, IMG_SIZE)

try:
    with st.spinner("Running prediction…"):
        if model_name == "FHD-HybridNet":
            probs, pred_idx, chosen_key, grad_model = run_fhd_ensemble(batch)
            cam_title = f"FHD-HybridNet (chosen: {chosen_key})"
        else:
            model = load_single_model(model_name)
            probs = model.predict(batch, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            grad_model = model
            cam_title = model_name

    pred_class = CLASS_NAMES[pred_idx]

    with st.spinner("Computing Grad-CAM…"):
        heatmap, feat_layer_name = gradcam_hooked(grad_model, batch, class_index=pred_idx)
        brain_mask = make_brain_mask_from_image(orig_img)
        heatmap2 = enhance_heatmap(
            heatmap,
            brain_mask=brain_mask,
            gamma=1.8,
            keep_percentile=85,
            keep_largest_blob=True
        )
        overlay = overlay_gradcam_on_image_fixed(heatmap2, orig_img, alpha=0.45)

except Exception as e:
    st.error("❌ Inference failed.")
    st.code(str(e))
    st.markdown(
        """
**If your error contains:**
- `dense expects 1 input but got 2`

Then your `.keras` model is not loading cleanly on Streamlit Cloud.  
✅ Fix WITHOUT retraining:

1) Open Colab  
2) Load your `.keras` model  
3) Re-save to `.h5`  
4) Upload `.h5` to Drive  
5) Update IDs in `app.py`  

If you want, paste your Colab conversion output links and I will update the IDs for `.h5` exactly.
"""
    )
    st.stop()

# ----------------------------
# 8) OUTPUT
# ----------------------------
st.markdown("---")
st.subheader("Prediction Output")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(orig_img)
axes[0].set_title(f"Original Image\nClass: {pred_class}")
axes[0].axis("off")

axes[1].imshow(overlay)
axes[1].set_title(f"Grad-CAM\n{cam_title}\nfeat: {feat_layer_name}")
axes[1].axis("off")

axes[2].barh(CLASS_NAMES, probs)
axes[2].set_xlim(0, 1)
axes[2].set_xlabel("Probability")
axes[2].set_title("Class Probabilities")

for i, _cls in enumerate(CLASS_NAMES):
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
