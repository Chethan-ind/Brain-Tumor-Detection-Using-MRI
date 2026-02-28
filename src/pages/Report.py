import streamlit as st
import io, datetime
from PIL import Image

# Copy minimal generate_pdf logic here to avoid importing the whole app
# and to keep report generation self-contained on the Report page.

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

    is_t = result.get("is_tumor", False)
    rc   = "#dc2626" if is_t else "#16a34a"
    story.append(Paragraph(
        f"<b>Result: <font color='{rc}'>{result.get('label','—').upper()}</font></b>",
        ParagraphStyle("Res", fontSize=15, spaceAfter=6)))

    rdata = [
        ["Metric","Value"],
        ["Raw Model Probability",    f"{result.get('probability',0):.6f}"],
        ["Tumor Class Index",        f"{result.get('tumor_class_idx','-')}"],
        ["Confidence",               f"{result.get('confidence',0):.1f}%"],
        ["Classification Threshold", f"{result.get('threshold',0):.4f}"],
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

    if img:
        try:
            ibuf = io.BytesIO()
            img.resize((200,200)).save(ibuf, format="PNG"); ibuf.seek(0)
            story.append(Paragraph("<b>MRI Thumbnail:</b>", styles["Normal"]))
            story.append(Spacer(1,6)); story.append(RLImage(ibuf, width=5*cm, height=5*cm))
            story.append(Spacer(1,20))
        except Exception:
            pass

    story.append(Paragraph(
        "<font color='#64748b' size='7'>AI research tool only. Not for clinical use. "
        "Consult a qualified radiologist for medical interpretation.</font>",
        styles["Normal"]))
    doc.build(story)
    return buf.getvalue()


# --- REPORT PAGE ---
st.set_page_config(page_title="NeuroScan Report", page_icon="🧾")
st.title("Clinical Report")

# Read required keys from session_state
required_keys = [
    "patient_name", "age", "gender", "patient_id", "result", "confidence"
]
missing = [k for k in required_keys if not st.session_state.get(k)]

if missing:
    st.warning("No report data found. Run an analysis on the main page and click 'Generate Report'.")
    st.write("Missing session keys:", missing)
else:
    # Build patient_info and result dicts compatible with generate_pdf
    patient_info = {
        "name": st.session_state.get("patient_name"),
        "id": st.session_state.get("patient_id"),
        "age": st.session_state.get("age"),
        "gender": st.session_state.get("gender"),
        "scan": st.session_state.get("scan_type", "—"),
        "physician": st.session_state.get("physician", "—"),
        "notes": st.session_state.get("notes", ""),
    }
    result = st.session_state.get("last_result") or {"label": st.session_state.get("result"), "confidence": st.session_state.get("confidence", 0)}

    st.markdown("**Patient**")
    st.write(patient_info)
    st.markdown("**Diagnosis**")
    st.markdown(f"### {result.get('label','—')}")
    st.markdown(f"**Confidence:** {result.get('confidence', st.session_state.get('confidence'))}%")
    st.markdown(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Build PIL image if available
    pil_img = None
    if st.session_state.get('last_image_bytes'):
        try:
            pil_img = Image.open(io.BytesIO(st.session_state['last_image_bytes']))
            st.image(pil_img, caption="Uploaded MRI", use_column_width=True)
        except Exception:
            pil_img = None

    # Generate PDF in-memory
    if st.button("Generate & Download PDF"):
        with st.spinner("Building PDF…"):
            pdf_bytes = generate_pdf(patient_info, st.session_state.get('last_result', result), pil_img)
        if pdf_bytes:
            fname = f"NeuroScan_{patient_info.get('id','report')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name=fname, mime="application/pdf")
        else:
            st.error("Failed to create PDF. Install reportlab: pip install reportlab")
