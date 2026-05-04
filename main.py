from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
from PIL import Image
from io import BytesIO
from huggingface_hub import snapshot_download
import traceback
import base64

app = FastAPI(title="Brain Tumor AI API")

# =========================
# Hugging Face Config
# =========================
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "doha14/brain-tumor-models"

# Cache model download ONCE
MODEL_DIR = None

# =========================
# Global Models
# =========================
clf_model = None
seg_model = None

# =========================
# Custom Loss Functions
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
# LOAD MODELS (FIXED)
# =========================
def load_models():
    global clf_model, seg_model, MODEL_DIR

    print("Loading models...")

    # Download ONCE and reuse cache
    if MODEL_DIR is None:
        MODEL_DIR = snapshot_download(
            repo_id=REPO_ID,
            token=HF_TOKEN,
            cache_dir="/tmp/models"
        )

    try:
        if clf_model is None:
            clf_model = tf.keras.models.load_model(
                f"{MODEL_DIR}/final_brisc_classifier_v2.keras",
                compile=False
            )

        if seg_model is None:
            seg_model = tf.keras.models.load_model(
                f"{MODEL_DIR}/2D_unet_segmentation_model.keras",
                custom_objects={
                    "dice_coef": dice_coef,
                    "dice_loss": dice_loss,
                    "bce_dice_loss": bce_dice_loss
                },
                compile=False
            )

        print("Models loaded successfully")

    except Exception as e:
        print("MODEL LOAD FAILED:", str(e))
        raise e

# =========================
# STARTUP
# =========================
@app.on_event("startup")
def startup():
    load_models()

# =========================
# CONSTANTS
# =========================
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
CLASS_SIZE = (300, 300)
SEG_SIZE = (256, 256)

# =========================
# PREPROCESSING
# =========================
def preprocess_classification(img):
    img = cv2.resize(img, CLASS_SIZE)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_segmentation(img):
    img = cv2.resize(img, SEG_SIZE)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# =========================
# POSTPROCESSING
# =========================
def clean_mask(mask, threshold=0.5):
    return (mask > threshold).astype(np.uint8)

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
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {"status": "API running"}

# =========================
# PREDICT ENDPOINT
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        global clf_model, seg_model

        if clf_model is None or seg_model is None:
            load_models()

        # Read image
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        img = np.array(image)

        # =========================
        # Classification
        # =========================
        clf_input = preprocess_classification(img)
        preds = clf_model.predict(clf_input, verbose=0)[0]

        pred_idx = int(np.argmax(preds))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(preds[pred_idx])

        # =========================
        # Segmentation
        # =========================
        if pred_class == "no_tumor":
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        else:
            seg_input = preprocess_segmentation(img)
            seg_output = seg_model.predict(seg_input, verbose=0)

            if seg_output.shape[-1] == 1:
                seg_pred = seg_output[0, :, :, 0]
                mask_small = clean_mask(seg_pred)
            else:
                seg_pred = np.argmax(seg_output, axis=-1)[0]
                mask_small = (seg_pred > 0).astype(np.uint8)

            mask = cv2.resize(
                mask_small,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # =========================
        # Overlay
        # =========================
        overlay = create_overlay(img, mask)

        _, buffer = cv2.imencode(
            '.png',
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )

        img_base64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "class": pred_class,
            "confidence": confidence,
            "overlay_image": img_base64
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }
