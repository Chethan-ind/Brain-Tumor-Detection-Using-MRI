"""
hf_model.py — Minimal Hugging Face (TF) image-classification integration

Provides a small wrapper to load a TF image-classification model from
Hugging Face and run inference on PIL images. Optional: requires the
`transformers` package with TensorFlow support.

This module is intentionally lightweight and used as an alternate
fallback model inside the Streamlit app. It does not replace the
existing Keras model pipeline but offers an easy way to test HF models.
"""
import numpy as np
from PIL import Image

def _require_transformers():
    try:
        from transformers import TFViTForImageClassification, ViTImageProcessor
        return TFViTForImageClassification, ViTImageProcessor
    except Exception as e:
        raise ImportError("transformers (with TF support) is required for Hugging Face models: " + str(e))


def load_hf_tf_model(model_name: str = "google/vit-base-patch16-224"):
    """Load a TF Vision Transformer image-classification model and processor.

    Returns a dict: { 'model': model, 'processor': processor, 'input_size': size }
    """
    TFModel, Processor = _require_transformers()
    processor = Processor.from_pretrained(model_name)
    model = TFModel.from_pretrained(model_name)

    # Determine expected input size from the processor (fallback to 224)
    size = 224
    if hasattr(processor, "size") and isinstance(processor.size, dict):
        size = int(processor.size.get("shortest_edge", size))
    return {"model": model, "processor": processor, "input_size": size}


def predict_hf(model, processor, pil_image: Image.Image) -> float:
    """Run a single forward pass and return probability for the positive class.

    Works for binary classifiers or models with a single positive class.
    Returns a float in [0,1].
    """
    import tensorflow as tf

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    inputs = processor(images=pil_image, return_tensors="tf")
    outputs = model(**inputs)

    # Handle logits → probabilities
    logits = outputs.logits
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

    # If model has two classes, assume index 1 is 'tumor' positive class
    if probs.shape[0] == 1:
        return float(probs[0])
    if probs.shape[0] >= 2:
        return float(probs[1])
    return float(probs[0])
