"""
setup.py  —  Run this ONCE to prepare your project for the app.

What it does:
  1. Renames best_model_phase2.h5 → brain_tumor_model.h5  (or uses phase1)
  2. Creates optimal_threshold.npy with default 0.5
  3. Verifies everything is ready

Run from project ROOT:
    python src/setup.py
"""

import os, sys, shutil
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
# If run from src/, go one level up
if os.path.basename(ROOT) == "src":
    ROOT = os.path.dirname(ROOT)

print("=" * 52)
print("  NeuroScan AI — Project Setup")
print(f"  Root: {ROOT}")
print("=" * 52)

# ── Step 1: Find & rename model ──────────────────────
target = os.path.join(ROOT, "brain_tumor_model.h5")

if os.path.exists(target):
    size = os.path.getsize(target) / (1024*1024)
    print(f"\n✅ brain_tumor_model.h5 already exists ({size:.1f} MB) — skipping rename.")
else:
    sources = [
        os.path.join(ROOT, "best_model_phase2.h5"),
        os.path.join(ROOT, "best_model_phase1.h5"),
    ]
    copied = False
    for src in sources:
        if os.path.exists(src):
            size = os.path.getsize(src) / (1024*1024)
            print(f"\n📋 Found: {os.path.basename(src)} ({size:.1f} MB)")
            shutil.copy2(src, target)
            print(f"✅ Copied → brain_tumor_model.h5")
            copied = True
            break
    if not copied:
        print("\n❌ No model file found in project root!")
        print("   Expected one of:")
        for s in sources:
            print(f"     {s}")
        print("\n   Please train the model first or copy a .h5 file to the root folder.")
        sys.exit(1)

# ── Step 2: Create threshold file ────────────────────
thresh_path = os.path.join(ROOT, "optimal_threshold.npy")
if os.path.exists(thresh_path):
    val = float(np.load(thresh_path))
    print(f"\n✅ optimal_threshold.npy exists (threshold = {val:.4f}) — skipping.")
else:
    default = 0.5
    np.save(thresh_path, default)
    print(f"\n✅ Created optimal_threshold.npy  (value = {default})")
    print("   Tip: Re-train with train.py to get a ROC-tuned threshold.")

# ── Step 3: Verify ───────────────────────────────────
print("\n── Verification ─────────────────────────────────")
checks = {
    "brain_tumor_model.h5"  : os.path.join(ROOT, "brain_tumor_model.h5"),
    "optimal_threshold.npy" : os.path.join(ROOT, "optimal_threshold.npy"),
    "src/app.py"            : os.path.join(ROOT, "src", "app.py"),
    "src/model.py"          : os.path.join(ROOT, "src", "model.py"),
    "src/predict.py"        : os.path.join(ROOT, "src", "predict.py"),
}
all_ok = True
for name, path in checks.items():
    if os.path.exists(path):
        print(f"  ✅  {name}")
    else:
        print(f"  ❌  {name}  ← MISSING")
        all_ok = False

print("\n" + "=" * 52)
if all_ok:
    print("  🎉  Setup complete! You can now run:")
    print()
    print("       cd src")
    print("       streamlit run app.py")
    print()
else:
    print("  ⚠️  Some files are missing — check above.")
print("=" * 52)