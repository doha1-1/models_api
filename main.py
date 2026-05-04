from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import os

# 🔥 IMPORTANT: must be before TF import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
from PIL import Image
from io import BytesIO
from huggingface_hub import hf_hub_download
import traceback
import base64
import threading

app = FastAPI(title="Brain Tumor AI API")

# =========================
# CONFIG
# =========================
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "doha14/brain-tumor-models"

CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
CLASS_SIZE = (300, 300)
SEG_SIZE = (256, 256)

# =========================
# GLOBAL MODELS
# =========================
clf_model = None
seg_model = None
model_lock = threading.Lock()  # prevents double loading

# =========================
# LOSS FUNCTIONS
# =========================
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

bce = tf.keras.losses.BinaryCrossentropy()

def bce_dice_loss(y_true, y_pred):
    return bce(y_true, y_pred) + dice_loss(y_true, y_pred)

# =========================
# LOAD MODELS (SAFE + SINGLE LOAD)
# =========================
def load_models():
    global clf_model, seg_model

    with model_lock:
        if clf_model is not None and seg_model is not None:
            return  # already loaded

        print("Loading models...")

        clf_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="final_brisc_classifier_v2.keras",
            token=HF_TOKEN
        )

        seg_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="2D_unet_segmentation_model.keras",
            token=HF_TOKEN
        )

        clf_model = tf.keras.models.load_model(clf_path, compile=False)

        seg_model = tf.keras.models.load_model(
            seg_path,
            custom_objects={
                "dice_coef": dice_coef,
                "dice_loss": dice_loss,
                "bce_dice_loss": bce_dice_loss
            },
            compile=False
        )

        print("Models loaded successfully.")

# =========================
# STARTUP (SAFE)
# =========================
@app.on_event("startup")
def startup():
    try:
        load_models()
    except Exception as e:
        print("MODEL LOAD FAILED:", str(e))

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {
        "status": "running",
        "clf_loaded": clf_model is not None,
        "seg_loaded": seg_model is not None
    }

# =========================
# PREPROCESSING
# =========================
def preprocess(img, size):
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# =========================
# MASK PROCESSING
# =========================
def clean_mask(mask):
    return (mask > 0.5).astype(np.uint8)

def create_overlay(image, mask):
    overlay = image.copy()
    green = np.zeros_like(image)
    green[..., 1] = 255

    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (
        0.6 * overlay[mask_bool] + 0.4 * green[mask_bool]
    ).astype(np.uint8)

    return overlay

# =========================
# PREDICT
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        load_models()  # safety check

        # read image
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        img = np.array(image)

        # ================= CLASSIFICATION =================
        clf_input = preprocess(img, CLASS_SIZE)
        preds = clf_model.predict(clf_input, verbose=0)[0]

        idx = int(np.argmax(preds))
        pred_class = CLASS_NAMES[idx]
        confidence = float(preds[idx])

        # ================= SEGMENTATION =================
        if pred_class == "no_tumor":
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        else:
            seg_input = preprocess(img, SEG_SIZE)
            seg_output = seg_model.predict(seg_input, verbose=0)

            if seg_output.shape[-1] == 1:
                mask_small = clean_mask(seg_output[0, :, :, 0])
            else:
                mask_small = (np.argmax(seg_output, axis=-1)[0] > 0).astype(np.uint8)

            mask = cv2.resize(
                mask_small,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # ================= OVERLAY =================
        overlay = create_overlay(img, mask)

        _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "class": pred_class,
            "confidence": confidence,
            "overlay_image": img_base64
        }

    except Exception as e:
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }
