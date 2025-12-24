# app.py — FHD-HybridNet Brain Tumor MRI (Streamlit + GitHub-ready)
# FIXES:
# 1) Removes sample_images Drive ZIP logic (you will commit images to repo manually)
# 2) Adds robust model loader to fix Keras/TensorFlow deserialization mismatch
#    (common on Streamlit Cloud when "keras" (Keras 3) conflicts with tf.keras)
#
# IMPORTANT RECOMMENDATION (deployment):
# - Streamlit Cloud should run Python 3.10/3.11 for TensorFlow stability.
# - Do NOT use Python 3.13 for TensorFlow unless you know exactly what you're doing.
#
# ------------------------------------------------------------

import os

# Force TF to use legacy Keras if available (helps when standalone `keras` v3 conflicts)
# Must be set BEFORE importing tensorflow.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import io
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import gdown
import cv2
import datetime

# ------------------------------------------------------------
# 0) BASIC SETUP
# ------------------------------------------------------------
st.set_page_config(
    page_title="FHD-HybridNet Brain Tumor MRI",
    layout="wide"
)

IMG_SIZE = (224, 224)

# If your training used a different class order, update this to match training order.
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

SAMPLE_DIR = "sample_images"
os.makedirs(SAMPLE_DIR, exist_ok=True)

MODEL_CACHE_DIR = "models_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1) GOOGLE DRIVE MODEL LINKS (YOUR FILE IDs)
# ------------------------------------------------------------
DENSENET_ID  = "1IVbNJA_TKFsT9ZftiWyihxp6CYSoJxwd"  # model_a best_model.keras
MOBILENET_ID = "1MOXJSc3GuHoq4T7ZIPKNqyDF3r33_pml"  # model_b best_model.keras
RESNET_ID    = "1zgfjc0JTIe1Xg24rcWTT7zym9FL2MObF"  # model_c best_model.keras


def gdrive_direct_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?id={file_id}"


# ------------------------------------------------------------
# 2) MODEL DOWNLOAD + ROBUST LOAD (FIX FOR YOUR ERROR)
# ------------------------------------------------------------
def _download_model_if_needed(file_id: str, filename: str) -> str:
    local_path = os.path.join(MODEL_CACHE_DIR, filename)
    if not os.path.exists(local_path):
        url = gdrive_direct_url(file_id)
        try:
            gdown.download(url, local_path, quiet=True)
        except Exception as e:
            st.error(f"Failed to download {filename} from Google Drive.\nError: {e}")
            raise

    if not os.path.exists(local_path):
        raise FileNotFoundError(local_path)

    return local_path


def _try_load_model_robust(local_path: str):
    """
    Fixes the exact error you got:
    ValueError: Layer "dense" expects 1 input(s), but it received 2 input tensors.

    That error usually appears when:
    - Streamlit environment has standalone `keras` (Keras 3) and TF/keras mismatch,
    - or TF is running in a backend mode that changes deserialization behavior.

    Strategy:
    1) Normal tf.keras load
    2) tf.keras load with safe_mode=False (Keras 3 compatible signature)
    3) If TF_USE_LEGACY_KERAS works and tf_keras is present, load with tf_keras
    """
    # 1) normal
    try:
        return tf.keras.models.load_model(local_path, compile=False)
    except Exception:
        pass

    # 2) safe_mode=False (only available on newer stacks; safe to try)
    try:
        return tf.keras.models.load_model(local_path, compile=False, safe_mode=False)
    except TypeError:
        # older signature without safe_mode
        pass
    except Exception:
        pass

    # 3) legacy tf_keras (when TF_USE_LEGACY_KERAS=1)
    try:
        import tf_keras  # provided when TF uses legacy Keras
        return tf_keras.models.load_model(local_path, compile=False)
    except Exception:
        pass

    # If all fail, raise a helpful message
    raise RuntimeError(
        "Model load failed in this environment.\n\n"
        "Most common cause: TensorFlow/Keras version mismatch on Streamlit Cloud.\n"
        "Fix: Use Python 3.10/3.11 and pin TensorFlow version in requirements.txt.\n"
        "Also avoid installing standalone `keras` separately unless you know why."
    )


@st.cache_resource(show_spinner=False)
def load_single_model(model_name: str):
    if model_name == "DenseNet121":
        file_id = DENSENET_ID
        fname = "bt1_model_a_densenet_best.keras"
    elif model_name == "MobileNetV1":
        file_id = MOBILENET_ID
        fname = "bt1_model_b_mobilenet_best.keras"
    elif model_name == "ResNet50V2":
        file_id = RESNET_ID
        fname = "bt1_model_c_resnet_best.keras"
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    local_path = _download_model_if_needed(file_id, fname)
    model = _try_load_model_robust(local_path)
    return model


@st.cache_resource(show_spinner=False)
def load_all_base_models():
    dn = load_single_model("DenseNet121")
    mb = load_single_model("MobileNetV1")
    rn = load_single_model("ResNet50V2")
    return {"DenseNet121": dn, "MobileNetV1": mb, "ResNet50V2": rn}


# ------------------------------------------------------------
# 3) IMAGE HELPERS
# ------------------------------------------------------------
def load_image_from_file(file_or_path, img_size=IMG_SIZE):
    if isinstance(file_or_path, str):
        img = Image.open(file_or_path).convert("RGB")
    else:
        img = Image.open(file_or_path).convert("RGB")

    img_resized = img.resize(img_size)
    arr = np.asarray(img_resized).astype("float32") / 255.0
    batch = np.expand_dims(arr, axis=0)
    return img, batch


# ------------------------------------------------------------
# 4) YOUR FINALIZED Grad-CAM HELPERS (Keras-safe)
# ------------------------------------------------------------
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
    # Sequential([backbone, head...])
    if hasattr(seq_model, "layers") and len(seq_model.layers) >= 2:
        return seq_model.layers[0]
    return None


def pick_rank4_feature_layer(backbone, input_shape=(224, 224, 3)):
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

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_map = conv_out["val"][0]

    heatmap = tf.reduce_sum(conv_map * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), feat_layer_name


# ------------------------------------------------------------
# 5) FHD ENSEMBLE HELPERS
# ------------------------------------------------------------
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
    models_dict = load_all_base_models()
    probs, pred_idx, chosen_key = ensemble_predict_fhd_single(models_dict, img_batch)
    grad_model = models_dict[chosen_key]
    return probs, pred_idx, chosen_key, grad_model


# ------------------------------------------------------------
# 6) SIDEBAR CONTROLS
# ------------------------------------------------------------
st.sidebar.title("Controls")

model_name = st.sidebar.selectbox(
    "Select model",
    ["DenseNet121", "MobileNetV1", "ResNet50V2", "FHD-HybridNet"],
    index=3
)

source = st.sidebar.radio(
    "Choose image source",
    ["Upload MRI", "Sample gallery"],
    index=1
)

chosen_file = None

if source == "Upload MRI":
    uploaded = st.sidebar.file_uploader(
        "Upload a brain MRI image",
        type=["png", "jpg", "jpeg", "webp"]
    )
    if uploaded is not None:
        chosen_file = uploaded
else:
    gallery_files = sorted(
        [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    )
    if gallery_files:
        sample_name = st.sidebar.selectbox("Pick a sample image", gallery_files)
        chosen_file = os.path.join(SAMPLE_DIR, sample_name)
    else:
        st.sidebar.warning("No images found in sample_images/ (add images to repo).")

run_button = st.sidebar.button("▶ Run prediction")

# ------------------------------------------------------------
# 7) MAIN HEADER
# ------------------------------------------------------------
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
    st.subheader("Selected options")
    st.write(f"**Model:** {model_name}")
    st.write(f"**Source:** {source}")
    if isinstance(chosen_file, str):
        st.write(f"**Image path:** `{chosen_file}`")
    elif chosen_file is None:
        st.write("_No image selected yet._")
    else:
        st.write(f"**Uploaded file:** `{chosen_file.name}`")

if not run_button:
    st.stop()

if chosen_file is None:
    st.error("Please upload or select an image first.")
    st.stop()

# ------------------------------------------------------------
# 8) RUN INFERENCE
# ------------------------------------------------------------
orig_img, batch = load_image_from_file(chosen_file, IMG_SIZE)

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
    heatmap2 = enhance_heatmap(heatmap, brain_mask=brain_mask, gamma=1.8, keep_percentile=85, keep_largest_blob=True)
    overlay = overlay_gradcam_on_image_fixed(heatmap2, orig_img, alpha=0.45)

# ------------------------------------------------------------
# 9) OUTPUT
# ------------------------------------------------------------
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
