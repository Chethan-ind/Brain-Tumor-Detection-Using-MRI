"""evaluate_testset.py
Run inference over `dataset/Testing` and report per-class metrics.

Usage: run from repository root (it imports `src.predict` which loads the model).

This script prints a confusion matrix and saves a small CSV of misclassified samples.
"""
import os
import csv
from collections import defaultdict

# Ensure src is on path when executed from repo root
_SRC = os.path.dirname(os.path.abspath(__file__))
import sys
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from predict import predict_image

ROOT = os.path.dirname(_SRC)
TEST_DIR = os.path.join(ROOT, "dataset", "Testing")

POSITIVE_FOLDERS = {"glioma_tumor", "meningioma_tumor", "pituitary_tumor"}

def is_positive_folder(name: str) -> bool:
    n = name.lower()
    if n in ("no_tumor", "no_tumour", "no-tumor", "none"):
        return False
    return any(p in n for p in ("tumor","tumour","glioma","meningioma","pituitary"))

def main():
    if not os.path.exists(TEST_DIR):
        print("Testing directory not found:", TEST_DIR)
        return

    stats = defaultdict(int)
    misclassified = []

    total = 0
    for cls in sorted(os.listdir(TEST_DIR)):
        cls_path = os.path.join(TEST_DIR, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in sorted(os.listdir(cls_path)):
            if not fname.lower().endswith(('.jpg','.jpeg','.png')):
                continue
            total += 1
            path = os.path.join(cls_path, fname)
            try:
                res = predict_image(path, use_tta=False, save_gradcam=False)
            except Exception as e:
                print(f"Error predicting {path}: {e}")
                continue

            pred_positive = bool(res.get("is_tumor"))
            true_positive = is_positive_folder(cls)

            if true_positive and pred_positive:
                stats['TP'] += 1
            elif true_positive and not pred_positive:
                stats['FN'] += 1
                misclassified.append((path, 'FN', res))
            elif not true_positive and pred_positive:
                stats['FP'] += 1
                misclassified.append((path, 'FP', res))
            else:
                stats['TN'] += 1

    tp = stats['TP']; tn = stats['TN']; fp = stats['FP']; fn = stats['FN']
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 2 * precision * recall / max(1e-9, (precision + recall))

    print("\nEvaluation summary")
    print("Total samples:", total)
    print(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"Accuracy: {acc:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")

    out_csv = os.path.join(_SRC, "misclassified.csv")
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["path","type","probability","confidence","threshold"])
        for path, ttype, res in misclassified:
            w.writerow([path, ttype, res.get('probability'), res.get('confidence'), res.get('threshold')])

    print(f"Saved {len(misclassified)} misclassified examples to: {out_csv}")

if __name__ == '__main__':
    main()
