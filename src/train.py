"""
train.py  —  Brain Tumor Detection | Fixed & Laptop-Safe
=========================================================
BUGS FIXED FROM PREVIOUS VERSION:
  1. Focal loss was producing NEGATIVE values → model could not learn
     → Replaced with standard binary_crossentropy (reliable, proven)
  2. AUC stuck at 0.5 (random) because loss was broken
  3. ROC curve crash (multiclass error) because model was outputting
     garbage predictions due to broken loss
  4. class_mode="binary" verified correct for 2-class sigmoid output
  5. Added loss sanity check before phase 2

Run: cd src && python train.py
"""

import os, gc, warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, roc_auc_score,
                              roc_curve, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ─────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────
IMG_SIZE    = 128
BATCH_SIZE  = 16
EPOCHS_HEAD = 20
EPOCHS_FINE = 20
LR_HEAD     = 1e-3
LR_FINE     = 1e-5
SEED        = 42

_SRC      = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.dirname(_SRC)
TRAIN_DIR = os.path.join(_ROOT, "dataset", "Training")
TEST_DIR  = os.path.join(_ROOT, "dataset", "Testing")
MODEL_OUT = os.path.join(_ROOT, "brain_tumor_model.h5")
THRESH_OUT= os.path.join(_ROOT, "optimal_threshold.npy")
P1_OUT    = os.path.join(_ROOT, "best_model_phase1.h5")
P2_OUT    = os.path.join(_ROOT, "best_model_phase2.h5")

tf.random.set_seed(SEED)
np.random.seed(SEED)

# Limit threads — prevents laptop overheating
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)


# ─────────────────────────────────────────────────────
#  DATA GENERATORS
# ─────────────────────────────────────────────────────
def build_generators():
    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode="nearest",
        validation_split=0.2,
    )
    val_aug  = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    test_aug = ImageDataGenerator(rescale=1./255)

    train_gen = train_aug.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",          # binary = single sigmoid output ✅
        subset="training",
        seed=SEED, shuffle=True)

    val_gen = val_aug.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        seed=SEED, shuffle=False)

    test_gen = test_aug.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False)

    return train_gen, val_gen, test_gen


# ─────────────────────────────────────────────────────
#  MODEL — MobileNetV2
# ─────────────────────────────────────────────────────
def build_model(trainable_base=False, unfreeze_layers=0):
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
#  CALLBACKS
# ─────────────────────────────────────────────────────
def get_callbacks(save_path: str, monitor="val_auc"):
    return [
        EarlyStopping(monitor=monitor, patience=6,
                      restore_best_weights=True, mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-8, verbose=1),
        ModelCheckpoint(save_path, monitor=monitor,
                        save_best_only=True, mode="max", verbose=1),
    ]


# ─────────────────────────────────────────────────────
#  SANITY CHECK — verify model outputs are sensible
# ─────────────────────────────────────────────────────
def sanity_check(model, val_gen, phase_name):
    print(f"\n🔬 Sanity check after {phase_name}...")
    val_gen.reset()
    batch_x, batch_y = next(iter(val_gen))
    preds = model.predict(batch_x, verbose=0).ravel()
    print(f"  Labels (first 10)     : {batch_y[:10].astype(int).tolist()}")
    print(f"  Predictions (first 10): {[round(float(p),3) for p in preds[:10]]}")
    print(f"  Pred range: [{preds.min():.4f}, {preds.max():.4f}]  mean={preds.mean():.4f}")
    if preds.max() - preds.min() < 0.01:
        print("  ⚠️  WARNING: Model outputting nearly constant values — still learning")
    else:
        print("  ✅ Model producing varied predictions — learning is working")
    return preds


# ─────────────────────────────────────────────────────
#  ROC THRESHOLD TUNING
# ─────────────────────────────────────────────────────
def find_optimal_threshold(model, val_gen):
    print("\n🔍 Finding optimal threshold via ROC...")
    val_gen.reset()

    y_true_list, y_prob_list = [], []
    steps = len(val_gen)
    for i in range(steps):
        bx, by = val_gen[i]
        preds  = model.predict(bx, verbose=0).ravel()
        y_true_list.extend(by.tolist())
        y_prob_list.extend(preds.tolist())

    y_true = np.array(y_true_list).astype(int)
    y_prob = np.array(y_prob_list)

    # Verify binary labels
    unique_labels = np.unique(y_true)
    print(f"  Unique labels in val set: {unique_labels}")
    if len(unique_labels) < 2:
        print("  ⚠️  Only one class in validation set — cannot compute ROC")
        default = 0.5
        np.save(THRESH_OUT, default)
        return default, 0.5

    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thr = float(thresholds[best_idx])

    np.save(THRESH_OUT, best_thr)

    print(f"  AUC-ROC          : {auc:.4f}")
    print(f"  Optimal Threshold: {best_thr:.4f}")
    print(f"  Sensitivity      : {tpr[best_idx]:.4f}")
    print(f"  Specificity      : {1-fpr[best_idx]:.4f}")
    print(f"  Saved → {THRESH_OUT}")

    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, color="royalblue", lw=2, label=f"AUC = {auc:.4f}")
    plt.scatter(fpr[best_idx], tpr[best_idx], color="red", s=100, zorder=5,
                label=f"Best threshold = {best_thr:.3f}")
    plt.plot([0,1],[0,1],"k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Brain Tumor Detection")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(_ROOT, "roc_curve.png"), dpi=120); plt.close()
    return best_thr, auc


# ─────────────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────────────
def evaluate(model, test_gen, threshold, class_indices):
    print("\n📈 Final evaluation on test set...")
    test_gen.reset()

    y_true_list, y_prob_list = [], []
    for i in range(len(test_gen)):
        bx, by = test_gen[i]
        preds  = model.predict(bx, verbose=0).ravel()
        y_true_list.extend(by.tolist())
        y_prob_list.extend(preds.tolist())

    y_true = np.array(y_true_list).astype(int)
    y_prob = np.array(y_prob_list)
    y_pred = (y_prob >= threshold).astype(int)

    idx_to_name  = {v: k for k, v in class_indices.items()}
    target_names = [idx_to_name.get(0,"Class 0"), idx_to_name.get(1,"Class 1")]

    print(classification_report(y_true, y_pred, target_names=target_names))

    try:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"  TP (tumor found)   : {tp}")
        print(f"  TN (normal correct): {tn}")
        print(f"  FP (false alarm)   : {fp}")
        print(f"  FN (tumor missed!) : {fn}")
        print(f"  Sensitivity: {tp/(tp+fn+1e-8):.4f}")
        print(f"  Specificity: {tn/(tn+fp+1e-8):.4f}")
    except Exception as e:
        print(f"  (Could not compute confusion matrix: {e})")


# ─────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────
def main():
    print("=" * 58)
    print("  Brain Tumor Detection — Fixed Training Pipeline")
    print("  MobileNetV2 | 128×128 | BCE Loss | Batch=16")
    print("=" * 58)

    for d in [TRAIN_DIR, TEST_DIR]:
        if not os.path.exists(d):
            print(f"\n❌ Not found: {d}"); return

    # ── Generators ──────────────────────────────────
    train_gen, val_gen, test_gen = build_generators()

    # ── Class mapping ────────────────────────────────
    print("\n" + "─"*45)
    print("  CLASS MAPPING")
    print("─"*45)
    for name, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
        is_tumor = "tumor" in name.lower() and "no" not in name.lower()
        tag = "  ← TUMOR ✅" if is_tumor else "  ← NO TUMOR"
        print(f"  Index {idx} = '{name}'{tag}")
    print("─"*45)

    tumor_class = 1
    for name, idx in train_gen.class_indices.items():
        if name.lower() in ("tumor","yes","1","positive","tumour"):
            tumor_class = idx
    print(f"  Tumor index: {tumor_class}")
    print(f"  Logic: sigmoid output >= threshold → TUMOR\n")

    labels  = train_gen.classes
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    cw      = dict(enumerate(weights))
    print(f"  Class weights : {cw}")
    print(f"  Train: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}\n")

    METRICS = [
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]

    # ══════════════════════════════════════════════════
    #  PHASE 1 — Head only, binary_crossentropy
    # ══════════════════════════════════════════════════
    print("=" * 58)
    print("  PHASE 1 — Training head (base frozen)")
    print("  Loss: binary_crossentropy (stable & proven)")
    print("=" * 58 + "\n")

    model = build_model(trainable_base=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_HEAD),
        loss="binary_crossentropy",   # ← FIXED: standard BCE, not focal loss
        metrics=METRICS,
    )

    trainable = sum(tf.size(v).numpy() for v in model.trainable_variables)
    print(f"  Trainable params: {trainable:,}\n")

    h1 = model.fit(
        train_gen,
        epochs=EPOCHS_HEAD,
        validation_data=val_gen,
        class_weight=cw,
        callbacks=get_callbacks(P1_OUT),
        verbose=1,
    )

    p1_auc = max(h1.history.get("val_auc", [0]))
    print(f"\n✅ Phase 1 done — Best val AUC: {p1_auc:.4f}")
    sanity_check(model, val_gen, "Phase 1")

    # ══════════════════════════════════════════════════
    #  PHASE 2 — Fine-tune top 20 layers
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 58)
    print("  PHASE 2 — Fine-tuning (top 20 layers unfrozen)")
    print("  Loss: binary_crossentropy | LR: 1e-5")
    print("=" * 58 + "\n")

    gc.collect()
    tf.keras.backend.clear_session()

    model2 = build_model(trainable_base=True, unfreeze_layers=20)
    try:
        model2.load_weights(P1_OUT)
        print("  ✅ Loaded best Phase 1 weights\n")
    except Exception:
        model2.set_weights(model.get_weights())
        print("  ⚠️  Using current weights\n")

    model2.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FINE),
        loss="binary_crossentropy",   # ← same stable loss
        metrics=METRICS,
    )

    trainable2 = sum(tf.size(v).numpy() for v in model2.trainable_variables)
    print(f"  Trainable params: {trainable2:,}\n")

    h2 = model2.fit(
        train_gen,
        epochs=EPOCHS_FINE,
        validation_data=val_gen,
        class_weight=cw,
        callbacks=get_callbacks(P2_OUT),
        verbose=1,
    )

    p2_auc = max(h2.history.get("val_auc", [0]))
    print(f"\n✅ Phase 2 done — Best val AUC: {p2_auc:.4f}")
    sanity_check(model2, val_gen, "Phase 2")

    # ── Pick best phase ──────────────────────────────
    if p2_auc >= p1_auc:
        best_model, best_phase, best_auc = model2, "Phase 2", p2_auc
    else:
        best_model, best_phase, best_auc = model,  "Phase 1", p1_auc
    print(f"\n🏆 Best: {best_phase} (AUC={best_auc:.4f})")

    # ── Save ─────────────────────────────────────────
    best_model.save(MODEL_OUT)
    print(f"✅ Saved → {MODEL_OUT}")

    # ── ROC threshold ────────────────────────────────
    threshold, auc = find_optimal_threshold(best_model, val_gen)

    # ── Evaluation ───────────────────────────────────
    evaluate(best_model, test_gen, threshold, train_gen.class_indices)

    # ── History plot ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13,4))
    for ax, key, title in zip(axes, ["auc","loss"], ["AUC","Loss"]):
        ax.plot(h1.history[key],         label="P1 Train", color="#3b82f6")
        ax.plot(h1.history[f"val_{key}"],label="P1 Val",   color="#3b82f6", linestyle="--")
        ax.plot(h2.history[key],         label="P2 Train", color="#22c55e")
        ax.plot(h2.history[f"val_{key}"],label="P2 Val",   color="#22c55e", linestyle="--")
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(_ROOT, "training_history.png"), dpi=120); plt.close()

    # ── Summary ──────────────────────────────────────
    print("\n" + "=" * 58)
    print("  🎉 TRAINING COMPLETE")
    print("=" * 58)
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  Threshold : {threshold:.4f}")
    print(f"  Model     : {MODEL_OUT}")
    print()
    print("  Class mapping (save this):")
    for name, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
        print(f"    {idx} = '{name}'")
    print()
    print("  Next: streamlit run app.py")
    print("=" * 58)


if __name__ == "__main__":
    main()