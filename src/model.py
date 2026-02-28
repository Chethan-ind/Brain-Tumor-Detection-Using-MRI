"""
model.py  —  Shared model utilities
Matches train.py exactly: MobileNetV2, 128×128
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import MobileNetV2

# ─────────────────────────────────────────────────────
#  AUTO PATH RESOLUTION
# ─────────────────────────────────────────────────────
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)

def _find_model() -> str:
    candidates = [
        os.path.join(_ROOT_DIR, "brain_tumor_model.h5"),
        os.path.join(_ROOT_DIR, "best_model_phase2.h5"),
        os.path.join(_ROOT_DIR, "best_model_phase1.h5"),
        os.path.join(_SRC_DIR,  "brain_tumor_model.h5"),
        os.path.join(_SRC_DIR,  "best_model_phase2.h5"),
        os.path.join(_SRC_DIR,  "best_model_phase1.h5"),
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"✅ Model: {os.path.basename(path)}")
            return path
    raise FileNotFoundError(
        "No model found. Run train.py first.\nLooked in:\n" +
        "\n".join(f"  {p}" for p in candidates))

def _find_threshold():
    for p in [os.path.join(_ROOT_DIR, "optimal_threshold.npy"),
              os.path.join(_SRC_DIR,  "optimal_threshold.npy")]:
        if os.path.exists(p):
            return p
    return None

# ─────────────────────────────────────────────────────
#  CONSTANTS  — must match train.py exactly
# ─────────────────────────────────────────────────────
IMG_SIZE          = 128          # ← matches train.py
DEFAULT_THRESHOLD = 0.5
GRADCAM_LAYER     = "Conv_1"     # last conv in MobileNetV2

MODEL_PATH     = _find_model()
THRESHOLD_PATH = _find_threshold()


# ─────────────────────────────────────────────────────
#  MODEL BUILDER
# ─────────────────────────────────────────────────────
def build_mobilenet_model(trainable_base=False, unfreeze_layers=0):
    base = MobileNetV2(
        include_top=False, weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = trainable_base
    if trainable_base and unfreeze_layers > 0:
        for layer in base.layers[:-unfreeze_layers]:
            layer.trainable = False

    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x   = base(inp, training=trainable_base)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dense(256, activation="relu",
                       kernel_regularizer=regularizers.l2(1e-4))(x)
    x   = layers.Dropout(0.4)(x)
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model(inp, out, name="BrainTumorMobileNet")


# ─────────────────────────────────────────────────────
#  PREPROCESSING  — must match training normalization
# ─────────────────────────────────────────────────────
def preprocess_image(img_input) -> np.ndarray:
    """
    Accepts: file path (str) | PIL Image | numpy array
    Returns: float32 (1, 128, 128, 3), values in [0, 1]
    """
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
        if img is None:
            raise ValueError(f"Cannot read: {img_input}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif hasattr(img_input, "convert"):      # PIL
        img = np.array(img_input.convert("RGB"))
    else:
        img = np.array(img_input)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0    # same as rescale=1/255
    return np.expand_dims(img, axis=0)      # (1, 128, 128, 3)


# ─────────────────────────────────────────────────────
#  GRAD-CAM
# ─────────────────────────────────────────────────────
def generate_gradcam(model, img_array, layer_name=GRADCAM_LAYER, alpha=0.45):
    def _last_conv(m):
        convs = []
        for l in m.layers:
            if isinstance(l, tf.keras.layers.Conv2D):
                convs.append(l.name)
            if hasattr(l, 'layers'):
                for sl in l.layers:
                    if isinstance(sl, tf.keras.layers.Conv2D):
                        convs.append(sl.name)
        return convs[-1] if convs else None

    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output])
    except ValueError:
        fallback = _last_conv(model)
        if not fallback:
            raise RuntimeError("No Conv2D layer found for Grad-CAM.")
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(fallback).output, model.output])

    with tf.GradientTape() as tape:
        inp_t          = tf.cast(img_array, tf.float32)
        conv_out, pred = grad_model(inp_t)
        loss           = pred[:, 0]

    grads        = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap      = (conv_out[0] @ pooled_grads[..., tf.newaxis]).numpy()
    heatmap      = np.squeeze(heatmap)
    heatmap      = np.maximum(heatmap, 0) / (heatmap.max() + 1e-8)

    h, w            = img_array.shape[1], img_array.shape[2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized),
                                         cv2.COLORMAP_JET)
    heatmap_rgb     = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    original        = np.uint8(img_array[0] * 255)
    return cv2.addWeighted(original, 1 - alpha, heatmap_rgb, alpha, 0)


# ─────────────────────────────────────────────────────
#  TEST-TIME AUGMENTATION
# ─────────────────────────────────────────────────────
def predict_with_tta(model, img_array, n_augments=8):
    preds = []
    for _ in range(n_augments):
        aug = img_array.copy()
        if np.random.rand() > 0.5:
            aug = aug[:, :, ::-1, :]
        angle  = np.random.uniform(-10, 10)
        M      = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), angle, 1.0)
        aug[0] = cv2.warpAffine(aug[0], M, (IMG_SIZE, IMG_SIZE))
        aug    = np.clip(aug * np.random.uniform(0.9, 1.1), 0, 1)
        preds.append(float(model.predict(aug, verbose=0)[0][0]))
    return float(np.mean(preds)), float(np.std(preds))


# ─────────────────────────────────────────────────────
#  LOAD THRESHOLD
# ─────────────────────────────────────────────────────
def load_threshold() -> float:
    if THRESHOLD_PATH:
        val = float(np.load(THRESHOLD_PATH))
        print(f"✅ Threshold: {val:.4f}")
        return val
    print(f"⚠️  No threshold file — using {DEFAULT_THRESHOLD}")
    return DEFAULT_THRESHOLD