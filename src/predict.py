"""
predict.py  —  CLI inference
Usage:
    python predict.py path/to/mri.jpg
    python predict.py path/to/mri.jpg --tta
    python predict.py path/to/mri.jpg --gradcam
    python predict.py path/to/mri.jpg --tta --gradcam
"""

import os, sys, argparse
import numpy as np
import cv2
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ── make sure src/ is on path when called from root ──
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from model import (
    preprocess_image, generate_gradcam,
    predict_with_tta, load_threshold,
    MODEL_PATH, GRADCAM_LAYER,
)

# ── Load model once ──────────────────────────────────
print("⏳ Loading model...", end=" ", flush=True)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅")

THRESHOLD = load_threshold()
CLASSES   = ["No Tumor", "Tumor"]


# ────────────────────────────────────────────────────
def predict_image(image_path: str,
                  use_tta=False,
                  save_gradcam=False) -> dict:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_array = preprocess_image(image_path)

    if use_tta:
        probability, uncertainty = predict_with_tta(model, img_array)
    else:
        probability = float(model.predict(img_array, verbose=0)[0][0])
        uncertainty = None

    # ✅ FIXED: sigmoid → threshold comparison (NOT argmax)
    is_tumor   = probability >= THRESHOLD
    label      = CLASSES[int(is_tumor)]
    confidence = probability if is_tumor else (1.0 - probability)

    result = {
        "label"      : label,
        "is_tumor"   : is_tumor,
        "probability": round(probability, 6),
        "confidence" : round(confidence * 100, 2),
        "uncertainty": round(uncertainty * 100, 2) if uncertainty else None,
        "threshold"  : round(THRESHOLD, 4),
    }

    if save_gradcam:
        try:
            heatmap  = generate_gradcam(model, img_array, GRADCAM_LAYER)
            out_path = os.path.splitext(image_path)[0] + "_gradcam.png"
            cv2.imwrite(out_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
            result["gradcam_path"] = out_path
            print(f"🗺️  Grad-CAM saved → {out_path}")
        except Exception as e:
            print(f"⚠️  Grad-CAM failed: {e}")

    return result


# ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Detection — CLI")
    parser.add_argument("image_path", help="Path to MRI image")
    parser.add_argument("--tta",     action="store_true",
                        help="Test-Time Augmentation (8 passes, slower but better)")
    parser.add_argument("--gradcam", action="store_true",
                        help="Save Grad-CAM heatmap next to input image")
    args = parser.parse_args()

    print("\n" + "═"*48)
    print("  🧠  Brain Tumor Detection System")
    print("═"*48)
    print(f"  Image    : {args.image_path}")
    print(f"  TTA      : {'Yes (8 passes)' if args.tta else 'No'}")
    print(f"  Grad-CAM : {'Yes' if args.gradcam else 'No'}")
    print(f"  Threshold: {THRESHOLD:.4f}")
    print("─"*48)

    result = predict_image(args.image_path,
                           use_tta=args.tta,
                           save_gradcam=args.gradcam)

    icon = "🔴" if result["is_tumor"] else "🟢"
    print(f"\n  {icon}  Prediction  : {result['label'].upper()}")
    print(f"  📊  Raw Prob    : {result['probability']:.6f}")
    print(f"  💯  Confidence  : {result['confidence']:.1f}%")
    if result["uncertainty"] is not None:
        print(f"  📉  Uncertainty : ±{result['uncertainty']:.1f}%")
    print("─"*48)

    if result["is_tumor"]:
        print("  ⚠️  Tumor indicators found.")
        print("      Please consult a qualified radiologist.")
    else:
        print("  ✅  No tumor indicators detected.")
        print("      Regular monitoring recommended.")

    print("\n  ⚠️  DISCLAIMER: Research tool — not for clinical use.")
    print("═"*48 + "\n")
    return result


if __name__ == "__main__":
    main()