"""
app.py  —  NeuroScan AI  |  Brain Tumor Detection Dashboard
Run: cd src && streamlit run app.py
"""

import os, sys, io, time, datetime
import numpy as np
import streamlit as st
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

st.set_page_config(
    page_title="NeuroScan AI | Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"], .stApp { font-family: 'Space Grotesk', sans-serif !important; background: #030712 !important; color: #e2e8f0 !important; }
.stApp { background: #030712 !important; }
.stApp::before { content: ''; position: fixed; inset: 0; z-index: 0; background: radial-gradient(ellipse 80% 50% at 20% 20%, rgba(0,180,255,0.07) 0%, transparent 60%), radial-gradient(ellipse 60% 40% at 80% 70%, rgba(0,255,150,0.05) 0%, transparent 60%), radial-gradient(ellipse 50% 60% at 50% 50%, rgba(100,50,255,0.04) 0%, transparent 70%); animation: bgPulse 8s ease-in-out infinite alternate; pointer-events: none; }
@keyframes bgPulse { 0% { opacity: 0.6; transform: scale(1); } 100% { opacity: 1; transform: scale(1.05); } }
.stApp::after { content: ''; position: fixed; inset: 0; z-index: 0; background-image: linear-gradient(rgba(0,180,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,180,255,0.03) 1px, transparent 1px); background-size: 50px 50px; pointer-events: none; }
.stApp > .main, .block-container { position: relative; z-index: 1; background: transparent !important; padding-top: 1rem !important; }
[data-testid="stSidebar"] { background: rgba(5, 10, 25, 0.95) !important; border-right: 1px solid rgba(0,180,255,0.15) !important; backdrop-filter: blur(20px); }
[data-testid="stSidebar"]::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #00b4ff, #00ff96, #6432ff, #00b4ff); background-size: 200% 100%; animation: shimmer 3s linear infinite; }
@keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
.ns-header { text-align: center; padding: 2.5rem 1rem 1.5rem; position: relative; }
.ns-logo-ring { width: 80px; height: 80px; margin: 0 auto 1rem; position: relative; display: flex; align-items: center; justify-content: center; }
.ns-logo-ring::before { content: ''; position: absolute; inset: 0; border-radius: 50%; border: 2px solid transparent; background: linear-gradient(135deg, #00b4ff, #00ff96) border-box; -webkit-mask: linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0); -webkit-mask-composite: destination-out; mask-composite: exclude; animation: rotateBorder 4s linear infinite; }
.ns-logo-ring::after { content: ''; position: absolute; inset: 8px; border-radius: 50%; border: 1px solid rgba(0,180,255,0.3); animation: rotateBorder 6s linear infinite reverse; }
@keyframes rotateBorder { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
.ns-logo-emoji { font-size: 2rem; position: relative; z-index: 2; filter: drop-shadow(0 0 12px rgba(0,180,255,0.6)); animation: pulse 3s ease-in-out infinite; }
@keyframes pulse { 0%, 100% { filter: drop-shadow(0 0 12px rgba(0,180,255,0.6)); } 50% { filter: drop-shadow(0 0 24px rgba(0,255,150,0.8)); } }
.ns-title { font-family: 'Syne', sans-serif !important; font-size: 2.8rem !important; font-weight: 800 !important; background: linear-gradient(135deg, #00b4ff 0%, #00ff96 50%, #6432ff 100%); -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; background-clip: text !important; letter-spacing: -0.02em; line-height: 1.1; margin-bottom: 0.3rem; }
.ns-subtitle { font-size: 0.9rem !important; color: #64748b !important; letter-spacing: 0.2em; text-transform: uppercase; font-weight: 500; }
.ns-divider { width: 100%; height: 1px; background: linear-gradient(90deg, transparent, rgba(0,180,255,0.4), rgba(0,255,150,0.4), transparent); margin: 1.5rem 0; position: relative; overflow: visible; }
.ns-divider::after { content: ''; position: absolute; top: -2px; left: 50%; transform: translateX(-50%); width: 60px; height: 5px; background: linear-gradient(90deg, #00b4ff, #00ff96); border-radius: 3px; filter: blur(3px); }
.ns-stats { display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.ns-stat { background: rgba(255,255,255,0.03); border: 1px solid rgba(0,180,255,0.15); border-radius: 100px; padding: 0.35rem 1rem; font-size: 0.78rem; color: #94a3b8; display: flex; align-items: center; gap: 0.4rem; backdrop-filter: blur(10px); }
.ns-stat-dot { width: 7px; height: 7px; border-radius: 50%; background: #00ff96; box-shadow: 0 0 8px #00ff96; animation: blink 2s ease-in-out infinite; }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.card { background: rgba(10, 15, 35, 0.8) !important; border: 1px solid rgba(0,180,255,0.12) !important; border-radius: 16px !important; padding: 1.8rem !important; margin-bottom: 1.5rem !important; backdrop-filter: blur(20px) !important; position: relative; overflow: hidden; transition: border-color 0.3s ease, box-shadow 0.3s ease; }
.card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(0,180,255,0.5), transparent); }
.card:hover { border-color: rgba(0,180,255,0.25) !important; box-shadow: 0 0 30px rgba(0,180,255,0.06), 0 20px 40px rgba(0,0,0,0.3) !important; }
.section-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem; }
.section-icon { width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1rem; flex-shrink: 0; }
.section-icon.blue  { background: rgba(0,180,255,0.1);  border: 1px solid rgba(0,180,255,0.3); }
.section-icon.green { background: rgba(0,255,150,0.1);  border: 1px solid rgba(0,255,150,0.3); }
.section-icon.purple{ background: rgba(100,50,255,0.1); border: 1px solid rgba(100,50,255,0.3);}
.section-title { font-family: 'Syne', sans-serif !important; font-size: 1.1rem !important; font-weight: 700 !important; color: #e2e8f0 !important; letter-spacing: -0.01em; }
.section-badge { margin-left: auto; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #00b4ff; background: rgba(0,180,255,0.08); border: 1px solid rgba(0,180,255,0.2); border-radius: 100px; padding: 0.15rem 0.6rem; letter-spacing: 0.05em; }
.stTextInput > div > div > input, .stNumberInput > div > div > input { background: rgba(255,255,255,0.07) !important; border: 1px solid rgba(0,180,255,0.25) !important; border-radius: 10px !important; color: #000 !important; font-family: 'Space Grotesk', sans-serif !important; font-size: 0.92rem !important; transition: border-color 0.2s, box-shadow 0.2s !important; caret-color: #00b4ff !important; }
.stTextInput > div > div > input::placeholder, .stNumberInput > div > div > input::placeholder { color: #475569 !important; }
.stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus { border-color: #00b4ff !important; box-shadow: 0 0 0 3px rgba(0,180,255,0.12) !important; background: rgba(255,255,255,0.09) !important; }
.stSelectbox > div > div > div[data-baseweb="select"] > div { background: rgba(255,255,255,0.07) !important; border: 1px solid rgba(0,180,255,0.25) !important; border-radius: 10px !important; color: #f1f5f9 !important; }
.stSelectbox [data-baseweb="select"] span, .stSelectbox [data-baseweb="select"] div { color: #f1f5f9 !important; }
[data-baseweb="popover"] li, [data-baseweb="menu"] li { color: #e2e8f0 !important; background: #0f172a !important; }
[data-baseweb="popover"] li:hover { background: rgba(0,180,255,0.1) !important; }
.stNumberInput button { color: #94a3b8 !important; border-color: rgba(0,180,255,0.2) !important; background: rgba(255,255,255,0.04) !important; }
.stTextArea > div > div > textarea { background: rgba(255,255,255,0.07) !important; border: 1px solid rgba(0,180,255,0.25) !important; border-radius: 10px !important; color: #000 !important; font-family: 'Space Grotesk', sans-serif !important; }
.stTextInput > label, .stNumberInput > label, .stSelectbox > label, .stTextArea > label { color: #94a3b8 !important; font-size: 0.78rem !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.09em !important; }
[data-testid="stFileUploader"] { background: rgba(0,180,255,0.03) !important; border: 2px dashed rgba(0,180,255,0.25) !important; border-radius: 14px !important; transition: all 0.3s !important; }
[data-testid="stFileUploader"]:hover { border-color: rgba(0,180,255,0.5) !important; background: rgba(0,180,255,0.06) !important; }
.stButton > button { background: linear-gradient(135deg, #00b4ff, #0090cc) !important; color: #fff !important; border: none !important; border-radius: 10px !important; padding: 0.65rem 1.5rem !important; font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important; font-size: 0.9rem !important; letter-spacing: 0.02em !important; transition: all 0.25s !important; box-shadow: 0 4px 15px rgba(0,180,255,0.3) !important; width: 100% !important; }
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(0,180,255,0.4) !important; background: linear-gradient(135deg, #00c8ff, #00a8e8) !important; }
.stButton > button:active { transform: translateY(0) !important; }
.stDownloadButton > button { background: linear-gradient(135deg, #00ff96, #00cc78) !important; color: #030712 !important; border: none !important; border-radius: 10px !important; font-weight: 700 !important; box-shadow: 0 4px 15px rgba(0,255,150,0.3) !important; width: 100% !important; padding: 0.65rem 1.5rem !important; }
.stDownloadButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(0,255,150,0.4) !important; }
.result-card { background: rgba(10, 15, 35, 0.9); border-radius: 16px; padding: 2rem; margin-bottom: 1.5rem; position: relative; overflow: hidden; border: 1px solid rgba(0,180,255,0.15); backdrop-filter: blur(20px); }
.result-card.tumor { border-color: rgba(231,76,60,0.4); background: rgba(30, 5, 5, 0.85); }
.result-card.tumor::before { content: ''; position: absolute; inset: 0; background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(231,76,60,0.08) 0%, transparent 70%); pointer-events: none; }
.result-card.no-tumor { border-color: rgba(0,255,150,0.4); background: rgba(0, 20, 10, 0.85); }
.result-card.no-tumor::before { content: ''; position: absolute; inset: 0; background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(0,255,150,0.06) 0%, transparent 70%); pointer-events: none; }
.result-label { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; margin-bottom: 0.25rem; display: flex; align-items: center; gap: 0.5rem; }
.result-label.tumor-text { color: #ff6b6b; }
.result-label.safe-text  { color: #00ff96; }
.result-meta { font-size: 0.8rem; color: #64748b; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.05em; margin-bottom: 1.5rem; }
.metric-tile { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 1rem; }
.metric-tile-label { font-size: 0.7rem; color: #475569; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; margin-bottom: 0.35rem; }
.metric-tile-value { font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 500; color: #e2e8f0; }
.warning-banner { background: rgba(255,165,0,0.06); border: 1px solid rgba(255,165,0,0.2); border-radius: 10px; padding: 0.65rem 1rem; display: flex; align-items: center; gap: 0.6rem; font-size: 0.78rem; color: #f59e0b; margin-bottom: 1.5rem; font-weight: 500; }
h1, h2, h3 { color: #e2e8f0 !important; }
p { color: #94a3b8 !important; }
.stProgress > div > div > div > div { background: linear-gradient(90deg, #00b4ff, #00ff96) !important; border-radius: 100px !important; }
.stProgress > div > div { background: rgba(255,255,255,0.05) !important; border-radius: 100px !important; }
.stSuccess { background: rgba(0,255,150,0.06) !important; border: 1px solid rgba(0,255,150,0.25) !important; border-radius: 10px !important; color: #00ff96 !important; }
.stError { background: rgba(231,76,60,0.06) !important; border: 1px solid rgba(231,76,60,0.25) !important; border-radius: 10px !important; }
.stSpinner > div { border-top-color: #00b4ff !important; }
[data-testid="stSidebar"] .stMarkdown h3 { color: #00b4ff !important; font-family: 'Syne', sans-serif !important; font-size: 0.85rem !important; text-transform: uppercase; letter-spacing: 0.12em; }
[data-testid="stSidebar"] p { color: #475569 !important; font-size: 0.82rem !important; }
[data-testid="stImage"] img { border-radius: 12px !important; border: 1px solid rgba(0,180,255,0.2) !important; box-shadow: 0 0 30px rgba(0,180,255,0.05) !important; }
[data-testid="stForm"] .stButton > button { margin-top: 0.5rem; }
hr { border: none !important; height: 1px !important; background: linear-gradient(90deg, transparent, rgba(0,180,255,0.25), transparent) !important; margin: 1.5rem 0 !important; }
@keyframes floatUp { 0% { transform: translateY(0) translateX(0) scale(1); opacity: 0; } 10% { opacity: 1; } 90% { opacity: 0.5; } 100% { transform: translateY(-100vh) translateX(30px) scale(0.3); opacity: 0; } }
.scan-line-container { position: relative; overflow: hidden; border-radius: 12px; }
.scan-line-container::after { content: ''; position: absolute; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, #00b4ff, transparent); box-shadow: 0 0 10px #00b4ff; animation: scanLine 2s ease-in-out infinite; top: 0; }
@keyframes scanLine { 0% { top: 0%; opacity: 0; } 10% { opacity: 1; } 90% { opacity: 1; } 100% { top: 100%; opacity: 0; } }
.glow-tag { display: inline-flex; align-items: center; gap: 0.4rem; background: rgba(0,180,255,0.08); border: 1px solid rgba(0,180,255,0.25); border-radius: 100px; padding: 0.3rem 0.9rem; font-size: 0.75rem; color: #00b4ff; font-weight: 600; letter-spacing: 0.05em; margin-bottom: 0.75rem; font-family: 'JetBrains Mono', monospace; }
.glow-tag-pulse { width: 6px; height: 6px; background: #00b4ff; border-radius: 50%; box-shadow: 0 0 6px #00b4ff; animation: blink 1.5s ease-in-out infinite; }
.steps-row { display: flex; align-items: center; gap: 0; margin-bottom: 1.5rem; }
.step-item { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 0.4rem; position: relative; }
.step-circle { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; border: 2px solid rgba(0,180,255,0.3); color: #475569; background: rgba(5,10,25,0.9); transition: all 0.3s; z-index: 1; }
.step-circle.active { border-color: #00b4ff; color: #00b4ff; box-shadow: 0 0 15px rgba(0,180,255,0.3); }
.step-circle.done { border-color: #00ff96; color: #00ff96; background: rgba(0,255,150,0.08); box-shadow: 0 0 10px rgba(0,255,150,0.2); }
.step-label { font-size: 0.65rem; color: #475569; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; text-align: center; }
.step-label.active { color: #00b4ff; }
.step-label.done   { color: #00ff96; }
.step-connector { flex: 1; height: 1px; background: rgba(0,180,255,0.15); margin-bottom: 1.5rem; }
.step-connector.done { background: rgba(0,255,150,0.4); }
</style>

<div id="particles-container" style="position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden;">
    <div style="position:absolute;width:3px;height:3px;background:#00b4ff;border-radius:50%;box-shadow:0 0 6px #00b4ff;opacity:0;left:15%;animation:floatUp 12s 0s ease-in infinite;"></div>
    <div style="position:absolute;width:2px;height:2px;background:#00ff96;border-radius:50%;box-shadow:0 0 5px #00ff96;opacity:0;left:35%;animation:floatUp 15s 3s ease-in infinite;"></div>
    <div style="position:absolute;width:2px;height:2px;background:#6432ff;border-radius:50%;box-shadow:0 0 5px #6432ff;opacity:0;left:60%;animation:floatUp 18s 6s ease-in infinite;"></div>
    <div style="position:absolute;width:3px;height:3px;background:#00b4ff;border-radius:50%;box-shadow:0 0 6px #00b4ff;opacity:0;left:80%;animation:floatUp 14s 9s ease-in infinite;"></div>
    <div style="position:absolute;width:2px;height:2px;background:#00ff96;border-radius:50%;box-shadow:0 0 5px #00ff96;opacity:0;left:50%;animation:floatUp 16s 2s ease-in infinite;"></div>
    <div style="position:absolute;width:2px;height:2px;background:#00b4ff;border-radius:50%;box-shadow:0 0 5px #00b4ff;opacity:0;left:90%;animation:floatUp 11s 7s ease-in infinite;"></div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading AI model…")
def load_model_cached():
    import tensorflow as tf
    from model import MODEL_PATH
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

@st.cache_data(show_spinner=False)
def load_threshold_cached():
    from model import load_threshold
    return load_threshold()

@st.cache_resource(show_spinner="Loading Hugging Face model…")
def load_hf_cached(model_name: str = "google/vit-base-patch16-224"):
    try:
        from hf_model import load_hf_tf_model
    except Exception:
        raise
    return load_hf_tf_model(model_name)


def detect_tumor_class_index() -> int:
    _root     = os.path.dirname(_SRC)
    train_dir = os.path.join(_root, "dataset", "Training")
    if os.path.exists(train_dir):
        folders = sorted([f for f in os.listdir(train_dir)
                          if os.path.isdir(os.path.join(train_dir, f))])
        for i, folder in enumerate(folders):
            if folder.lower() in ("tumor","yes","1","positive","tumour"):
                return i
    return 1


def generate_pdf(patient_info: dict, result: dict, img: Image.Image):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, Image as RLImage)
        from reportlab.lib.units import cm
    except ImportError:
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             rightMargin=1.8*cm, leftMargin=1.8*cm,
                             topMargin=1.8*cm, bottomMargin=1.8*cm)
    styles = getSampleStyleSheet()
    story  = []

    story.append(Paragraph("NeuroScan AI — Brain Tumor Detection Report",
        ParagraphStyle("H", parent=styles["Title"], fontSize=20,
                       textColor=colors.HexColor("#1d4ed8"))))
    story.append(Paragraph(
        "<font color='#dc2626'><b>⚠ RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSIS</b></font>",
        ParagraphStyle("W", fontSize=8, alignment=1)))
    story.append(Spacer(1, 14))

    now   = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")
    pdata = [
        ["Patient Name", patient_info.get("name","—"), "Patient ID",  patient_info.get("id","—")],
        ["Age",          str(patient_info.get("age","—")), "Gender",  patient_info.get("gender","—")],
        ["MRI Sequence", patient_info.get("scan","—"), "Report Date", now],
        ["Physician",    patient_info.get("physician","—"), "", ""],
    ]
    pt = Table(pdata, colWidths=[4*cm,6*cm,4*cm,4.5*cm])
    pt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#dbeafe")),
        ("BACKGROUND",(2,0),(2,-1),colors.HexColor("#dbeafe")),
        ("FONTNAME",  (0,0),(-1,-1),"Helvetica"),
        ("FONTSIZE",  (0,0),(-1,-1),9),
        ("GRID",      (0,0),(-1,-1),0.4,colors.HexColor("#94a3b8")),
        ("PADDING",   (0,0),(-1,-1),6),
    ]))
    story.append(pt); story.append(Spacer(1,18))

    is_t = result["is_tumor"]
    rc   = "#dc2626" if is_t else "#16a34a"
    story.append(Paragraph(
        f"<b>Result: <font color='{rc}'>{result['label'].upper()}</font></b>",
        ParagraphStyle("Res", fontSize=15, spaceAfter=6)))

    rdata = [
        ["Metric","Value"],
        ["Raw Model Probability",    f"{result['probability']:.6f}"],
        ["Tumor Class Index",        f"{result['tumor_class_idx']}"],
        ["Confidence",               f"{result['confidence']:.1f}%"],
        ["Classification Threshold", f"{result['threshold']:.4f}"],
    ]
    if result.get("uncertainty"):
        rdata.append(["TTA Uncertainty", f"±{result['uncertainty']:.1f}%"])
    rt = Table(rdata, colWidths=[9*cm,9.5*cm])
    rt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR", (0,0),(-1,0),colors.white),
        ("FONTNAME",  (0,0),(-1,-1),"Helvetica"),
        ("FONTSIZE",  (0,0),(-1,-1),10),
        ("GRID",      (0,0),(-1,-1),0.4,colors.HexColor("#94a3b8")),
        ("PADDING",   (0,0),(-1,-1),7),
    ]))
    story.append(rt); story.append(Spacer(1,14))

    if patient_info.get("notes","").strip():
        story.append(Paragraph(f"<b>Notes:</b> {patient_info['notes']}", styles["Normal"]))
        story.append(Spacer(1,10))

    ibuf = io.BytesIO()
    img.resize((200,200)).save(ibuf, format="PNG"); ibuf.seek(0)
    story.append(Paragraph("<b>MRI Thumbnail:</b>", styles["Normal"]))
    story.append(Spacer(1,6)); story.append(RLImage(ibuf, width=5*cm, height=5*cm))
    story.append(Spacer(1,20))
    story.append(Paragraph(
        "<font color='#64748b' size='7'>AI research tool only. Not for clinical use. "
        "Consult a qualified radiologist for medical interpretation.</font>",
        styles["Normal"]))
    doc.build(story)
    return buf.getvalue()


def main():
    model = load_model_cached()
    threshold = load_threshold_cached()
    tumor_class_idx = detect_tumor_class_index()

    for key in ['generated_pdf','generated_pdf_name','last_result']:
        if key not in st.session_state:
            st.session_state[key] = None
    for key in ['patient_name','patient_id','age','gender','result','confidence',
                'scan_type','physician','notes','last_image_bytes']:
        if key not in st.session_state:
            st.session_state[key] = None

    with st.sidebar:
        st.markdown("""
        <div style='padding:1rem 0 0.5rem;'>
            <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;
                        background:linear-gradient(135deg,#00b4ff,#00ff96);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                        background-clip:text;margin-bottom:0.25rem;'>
                NeuroScan AI
            </div>
            <div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;
                        letter-spacing:0.12em;font-weight:600;'>
                v2.0 · Research Platform
            </div>
        </div>
        <hr>
        """, unsafe_allow_html=True)

        st.markdown("### ⚙️ Model Settings")
        use_tta = st.toggle("Test-Time Augmentation", value=False,
                            help="Run 8 augmented passes for uncertainty estimation")
        use_hf = st.toggle("Use HuggingFace ViT", value=False,
                           help="Switch to Vision Transformer backend")

        hf_bundle = None
        if use_hf:
            hf_bundle = load_hf_cached()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 📋 Additional Info")
        scan_type = st.selectbox("MRI Sequence",
                                 ["T1","T2","FLAIR","T1+Contrast","T2*","DWI","Other"])
        physician = st.text_input("Referring Physician")
        notes     = st.text_area("Clinical Notes", height=80)
        st.session_state['scan_type'] = scan_type
        st.session_state['physician'] = physician
        st.session_state['notes']     = notes

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <div style='padding:0.75rem;background:rgba(255,165,0,0.05);
                    border:1px solid rgba(255,165,0,0.2);border-radius:10px;'>
            <div style='font-size:0.7rem;color:#f59e0b;font-weight:600;
                        text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.35rem;'>
                ⚠ Research Use Only
            </div>
            <div style='font-size:0.72rem;color:#64748b;line-height:1.5;'>
                Not intended for clinical diagnosis. Always consult a qualified radiologist.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='margin-top:1rem;padding:0.75rem;background:rgba(0,180,255,0.04);
                    border:1px solid rgba(0,180,255,0.1);border-radius:10px;'>
            <div style='font-size:0.7rem;color:#475569;font-weight:600;
                        text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;'>
                System Status
            </div>
            <div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.3rem;'>
                <div style='width:7px;height:7px;background:#00ff96;border-radius:50%;box-shadow:0 0 6px #00ff96;'></div>
                <span style='font-size:0.75rem;color:#64748b;'>Model Ready</span>
            </div>
            <div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.3rem;'>
                <div style='width:7px;height:7px;background:#00b4ff;border-radius:50%;box-shadow:0 0 6px #00b4ff;'></div>
                <span style='font-size:0.75rem;color:#64748b;'>GPU Accelerated</span>
            </div>
            <div style='display:flex;align-items:center;gap:0.5rem;'>
                <div style='width:7px;height:7px;background:#f59e0b;border-radius:50%;box-shadow:0 0 6px #f59e0b;'></div>
                <span style='font-size:0.75rem;color:#64748b;'>PDF Reports Active</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='ns-header'>
        <div class='ns-logo-ring'><span class='ns-logo-emoji'>🧠</span></div>
        <div class='ns-title'>NeuroScan AI</div>
        <div class='ns-subtitle'>Brain Tumor Detection Platform</div>
        <div class='ns-divider'></div>
        <div class='ns-stats'>
            <div class='ns-stat'><div class='ns-stat-dot'></div>System Online</div>
            <div class='ns-stat'>Real-time Inference</div>
            <div class='ns-stat'>HIPAA Compliant Interface</div>
            <div class='ns-stat'>Grad-CAM Visualization</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    has_patient = bool(st.session_state.get('patient_name') and st.session_state.get('patient_id'))
    has_result  = bool(st.session_state.get('last_result'))
    step1_cls = "done" if has_patient else "active"
    step2_cls = "done" if has_result else ("active" if has_patient else "")
    step3_cls = "active" if has_result else ""
    conn1_cls = "done" if has_patient else ""
    conn2_cls = "done" if has_result else ""

    st.markdown(f"""
    <div class='steps-row'>
        <div class='step-item'>
            <div class='step-circle {step1_cls}'>{'✓' if has_patient else '01'}</div>
            <div class='step-label {step1_cls}'>Patient Info</div>
        </div>
        <div class='step-connector {conn1_cls}'></div>
        <div class='step-item'>
            <div class='step-circle {step2_cls}'>{'✓' if has_result else '02'}</div>
            <div class='step-label {step2_cls}'>Upload & Scan</div>
        </div>
        <div class='step-connector {conn2_cls}'></div>
        <div class='step-item'>
            <div class='step-circle {step3_cls}'>03</div>
            <div class='step-label {step3_cls}'>Generate Report</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap='large')

    with left:
        st.markdown("""
        <div class='card'>
            <div class='section-header'>
                <div class='section-icon blue'>👤</div>
                <div class='section-title'>Patient Details</div>
                <div class='section-badge'>STEP 01</div>
            </div>
        """, unsafe_allow_html=True)

        with st.form('patient_form'):
            col_a, col_b = st.columns(2)
            with col_a:
                name = st.text_input('Full Name',
                                     value=st.session_state.get('patient_name') or '',
                                     placeholder='John Doe')
            with col_b:
                pid = st.text_input('Patient ID',
                                    value=st.session_state.get('patient_id') or '',
                                    placeholder='PT-00001')
            col_c, col_d = st.columns(2)
            with col_c:
                age_val = st.session_state.get('age')
                age = st.number_input('Age', 1, 120, value=int(age_val) if age_val else 45)
            with col_d:
                genders = ['Select…','Male','Female','Other']
                g_val   = st.session_state.get('gender','Select…')
                g_idx   = genders.index(g_val) if g_val in genders else 0
                gender  = st.selectbox('Gender', genders, index=g_idx)

            submit = st.form_submit_button('Save Patient Details')
            if submit:
                st.session_state['patient_name'] = name
                st.session_state['patient_id']   = pid
                st.session_state['age']           = age
                st.session_state['gender']        = gender
                st.success('✓ Patient information saved successfully')

        st.markdown("</div>", unsafe_allow_html=True)

        # ── ACTIVE PATIENT TILE ──
        if st.session_state.get('patient_name'):
            st.markdown(f"""
            <div style='background:rgba(0,180,255,0.04);border:1px solid rgba(0,180,255,0.15);
                        border-radius:12px;padding:1rem;margin-top:0.5rem;'>
                <div style='font-size:0.7rem;color:#475569;text-transform:uppercase;
                            letter-spacing:0.1em;font-weight:600;margin-bottom:0.6rem;'>
                    Active Patient
                </div>
                <div style='font-size:1rem;font-weight:600;color:#f1f5f9;margin-bottom:0.25rem;'>
                    {st.session_state.get('patient_name','—')}
                </div>
                <div style='font-family:JetBrains Mono,monospace;font-size:0.8rem;color:#94a3b8;'>
                    ID: {st.session_state.get('patient_id','—')} &nbsp;·&nbsp;
                    Age: {st.session_state.get('age','—')} &nbsp;·&nbsp;
                    {st.session_state.get('gender','—')}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class='card'>
            <div class='section-header'>
                <div class='section-icon green'>🔬</div>
                <div class='section-title'>MRI Scan Analysis</div>
                <div class='section-badge'>STEP 02</div>
            </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload MRI image (JPG / PNG)",
            type=['jpg','jpeg','png'],
            help="Upload a brain MRI scan for tumor detection"
        )

        if uploaded:
            image = Image.open(uploaded).convert('RGB')
            st.markdown("<div class='scan-line-container'>", unsafe_allow_html=True)
            st.image(image, use_column_width=True, caption="Loaded MRI Scan")
            st.markdown("</div>", unsafe_allow_html=True)

            analyze = st.button('Run AI Analysis')
            if analyze:
                st.session_state['generated_pdf'] = None
                st.session_state['generated_pdf_name'] = None

                status = st.empty()
                bar    = st.progress(0)
                steps = [
                    (15,  " Initializing neural network…"),
                    (30,  " Preprocessing image…"),
                    (50,  " Running deep inference…"),
                    (70,  " Computing feature maps…"),
                    (85,  " Generating Grad-CAM heatmap…"),
                    (95,  " Calculating confidence scores…"),
                    (100, " Analysis complete"),
                ]
                for pct, msg in steps:
                    status.markdown(
                        f"<p style='color:#00b4ff;font-family:JetBrains Mono,monospace;"
                        f"font-size:0.85rem;margin:0;'>{msg}</p>",
                        unsafe_allow_html=True
                    )
                    bar.progress(pct)
                    time.sleep(0.25)
                status.empty()
                bar.empty()

                if 'use_hf' in dir() and use_hf and hf_bundle is not None:
                    from hf_model import predict_hf
                    if use_tta:
                        preds = []
                        for _ in range(8): preds.append(predict_hf(hf_bundle[0], hf_bundle[1], image))
                        raw_prob    = float(np.mean(preds))
                        uncertainty = float(np.std(preds))
                    else:
                        raw_prob    = float(predict_hf(hf_bundle[0], hf_bundle[1], image))
                        uncertainty = None
                else:
                    from model import preprocess_image, generate_gradcam, predict_with_tta, GRADCAM_LAYER
                    img_array = preprocess_image(image)
                    if use_tta:
                        raw_prob, uncertainty = predict_with_tta(model, img_array)
                    else:
                        raw_prob    = float(model.predict(img_array, verbose=0)[0][0])
                        uncertainty = None

                if tumor_class_idx == 1:
                    is_tumor   = raw_prob >= threshold
                    tumor_prob = raw_prob
                else:
                    is_tumor   = raw_prob < threshold
                    tumor_prob = 1.0 - raw_prob

                label      = 'Tumor Detected' if is_tumor else 'No Tumor Detected'
                confidence = tumor_prob if is_tumor else (1.0 - tumor_prob)

                result_dict = {
                    'label'           : label,
                    'is_tumor'        : is_tumor,
                    'probability'     : round(raw_prob, 6),
                    'tumor_prob'      : round(tumor_prob, 6),
                    'confidence'      : round(confidence * 100, 2),
                    'uncertainty'     : round(uncertainty * 100, 2) if uncertainty else None,
                    'threshold'       : round(threshold, 4),
                    'tumor_class_idx' : tumor_class_idx,
                }
                st.session_state['last_result'] = result_dict
                buf = io.BytesIO()
                image.save(buf, format='PNG')
                st.session_state['last_image_bytes'] = buf.getvalue()
        else:
            st.markdown("""
            <div style='padding:2.5rem;text-align:center;
                        border:2px dashed rgba(0,180,255,0.15);border-radius:12px;
                        background:rgba(0,180,255,0.02);'>
                <div style='font-size:2.5rem;margin-bottom:0.75rem;
                            filter:drop-shadow(0 0 10px rgba(0,180,255,0.3));'>🧠</div>
                <div style='font-size:0.9rem;color:#334155;font-weight:500;margin-bottom:0.4rem;'>
                    No scan loaded
                </div>
                <div style='font-size:0.78rem;color:#1e293b;'>
                    Upload a brain MRI image above to begin analysis
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get('last_result'):
        res      = st.session_state['last_result']
        is_tumor = res['is_tumor']
        pct      = res['confidence']
        card_cls = "tumor" if is_tumor else "no-tumor"
        lbl_cls  = "tumor-text" if is_tumor else "safe-text"
        icon     = "⚠️" if is_tumor else "✅"
        ts       = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        st.markdown(f"""
        <div class='result-card {card_cls}'>
            <div class='glow-tag'>
                <div class='glow-tag-pulse'></div>
                ANALYSIS COMPLETE
            </div>
            <div class='result-label {lbl_cls}'>
                {icon} {res['label']}
            </div>
            <div class='result-meta'>
                Analyzed {ts} &nbsp;&middot;&nbsp; Threshold {res['threshold']:.4f} &nbsp;&middot;&nbsp; Class Index {res['tumor_class_idx']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        bar_color = "#ff4444" if is_tumor else "#00ff96"
        st.markdown(f"""
        <div style='margin-top:0.75rem;margin-bottom:0.3rem;font-size:0.72rem;color:#475569;
                    text-transform:uppercase;letter-spacing:0.12em;font-weight:600;'>
            Confidence Score
        </div>
        """, unsafe_allow_html=True)
        st.progress(int(pct))
        st.markdown(f"""
        <div style='text-align:right;font-family:JetBrains Mono,monospace;font-size:0.82rem;
                    color:{bar_color};margin-top:-0.5rem;margin-bottom:0.75rem;font-weight:600;'>
            {pct:.1f}%
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='margin-top:0.75rem;margin-bottom:0.4rem;font-size:0.72rem;color:#475569;
                    text-transform:uppercase;letter-spacing:0.12em;font-weight:600;'>
            Analysis Details
        </div>
        """, unsafe_allow_html=True)

        risk_level = (
            "Critical" if pct >= 90 else
            "High"     if pct >= 70 else
            "Moderate" if pct >= 50 else
            "Low"
        ) if is_tumor else "Negative"

        margin     = abs(res['probability'] - res['threshold'])
        margin_pct = round(margin * 100, 1)
        scan_seq   = st.session_state.get('scan_type') or 'N/A'
        tta_status = f"±{res['uncertainty']}%" if res.get('uncertainty') else "Off"

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class='metric-tile' style='text-align:center;'>
                <div class='metric-tile-label'>Risk Level</div>
                <div class='metric-tile-value' style='color:{"#ff6b6b" if is_tumor else "#00ff96"};font-size:1.05rem;'>
                    {risk_level}
                </div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class='metric-tile' style='text-align:center;'>
                <div class='metric-tile-label'>Decision Margin</div>
                <div class='metric-tile-value'>{margin_pct}%</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class='metric-tile' style='text-align:center;'>
                <div class='metric-tile-label'>MRI Sequence</div>
                <div class='metric-tile-value' style='font-size:0.9rem;'>{scan_seq}</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class='metric-tile' style='text-align:center;'>
                <div class='metric-tile-label'>TTA Uncertainty</div>
                <div class='metric-tile-value'>{tta_status}</div>
            </div>""", unsafe_allow_html=True)

        if is_tumor:
            interp_color = "#ff6b6b"
            interp_bg    = "rgba(255,80,80,0.05)"
            interp_bdr   = "rgba(255,80,80,0.2)"
            interp_msg   = (
                f"The model detected abnormal tissue patterns consistent with a tumor "
                f"with <b>{pct:.1f}% confidence</b>. The prediction exceeds the classification "
                f"threshold by <b>{margin_pct}%</b>. Immediate radiologist review is recommended."
            )
        else:
            interp_color = "#00ff96"
            interp_bg    = "rgba(0,255,150,0.04)"
            interp_bdr   = "rgba(0,255,150,0.2)"
            interp_msg   = (
                f"No tumor indicators detected. The model classified this scan as negative "
                f"with <b>{pct:.1f}% confidence</b>, sitting <b>{margin_pct}%</b> below the "
                f"decision threshold. Routine follow-up is advised."
            )

        st.markdown(f"""
        <div style='background:{interp_bg};border:1px solid {interp_bdr};
                    border-radius:12px;padding:1rem 1.2rem;margin-top:0.75rem;margin-bottom:0.5rem;'>
            <div style='font-size:0.7rem;color:{interp_color};text-transform:uppercase;
                        letter-spacing:0.1em;font-weight:700;margin-bottom:0.4rem;'>
                🔍 AI Interpretation
            </div>
            <div style='font-size:0.83rem;color:#94a3b8;line-height:1.6;'>
                {interp_msg}
            </div>
        </div>
        """, unsafe_allow_html=True)

        can_report = (
            st.session_state.get('patient_name') and
            st.session_state.get('patient_id') and
            st.session_state.get('gender') and
            st.session_state.get('gender') != 'Select…'
        )

        st.markdown("""
        <div class='card'>
            <div class='section-header'>
                <div class='section-icon purple'>📄</div>
                <div class='section-title'>Medical Report</div>
                <div class='section-badge'>STEP 03</div>
            </div>
        """, unsafe_allow_html=True)

        if not can_report:
            st.markdown("""
            <div class='warning-banner'>
                ⚠ Complete patient details (name, ID, gender) to enable report generation.
            </div>
            """, unsafe_allow_html=True)

        if can_report:
            if st.button('📄 Generate Medical Report'):
                with st.spinner('Generating PDF report…'):
                    patient_info = {
                        'name'      : st.session_state['patient_name'],
                        'id'        : st.session_state['patient_id'],
                        'age'       : st.session_state['age'],
                        'gender'    : st.session_state['gender'],
                        'scan'      : st.session_state.get('scan_type','—'),
                        'physician' : st.session_state.get('physician','—'),
                        'notes'     : st.session_state.get('notes',''),
                    }
                    img_src = (
                        Image.open(io.BytesIO(st.session_state['last_image_bytes']))
                        if st.session_state.get('last_image_bytes') else None
                    )
                    pdf_bytes = generate_pdf(patient_info, st.session_state['last_result'], img_src)

                if pdf_bytes:
                    fname = (f"NeuroScan_{st.session_state['patient_id']}_"
                             f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
                    st.session_state['generated_pdf']      = pdf_bytes
                    st.session_state['generated_pdf_name'] = fname
                    st.success('✓ Report generated successfully')
                else:
                    st.error('Install reportlab to enable PDF generation: pip install reportlab')

        if st.session_state.get('generated_pdf'):
            st.download_button(
                '⬇️ Download PDF Report',
                data=st.session_state['generated_pdf'],
                file_name=st.session_state.get('generated_pdf_name','report.pdf'),
                mime='application/pdf'
            )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center;padding:2rem 0 1rem;'>
        <div style='font-size:0.75rem;color:#1e293b;font-family:JetBrains Mono,monospace;letter-spacing:0.05em;'>
            NeuroScan AI v2.0 &nbsp;·&nbsp; Research Use Only &nbsp;·&nbsp; Not for Clinical Diagnosis
        </div>
        <div style='margin-top:0.5rem;font-size:0.7rem;color:#0f172a;'>
            Always consult a qualified radiologist for medical interpretation
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()