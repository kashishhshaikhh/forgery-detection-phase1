import base64
import html
import io
import os
import tempfile

import streamlit as st
from PIL import Image

from casia_predict import predict_casia
from cifake_predict import predict_cifake

# Scores between LOW and HIGH are treated as "not confident" (presentation choice,
# not a validated forensic threshold).
LOW, HIGH = 0.30, 0.70
MAX_MB = 10

st.set_page_config(page_title="Image Integrity Analyzer", page_icon="◈", layout="wide")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
:root{--plum-dark:#3B1033;--plum:#641B50;--plum-bright:#8B2D70;--oatmeal:#E8DDCE;--cream:#FBF7F0;--cream-dark:#F1E8DC;--text:#321F2D;--muted:#75656D;--real:#2F7655;--real-light:#E2F0E8;--berry:#8B294A;--berry-light:#F3E2E7;--border:#D8C9BD;--shadow:0 18px 50px rgba(59,16,51,.10);}
html,body,.stApp,[class*="st-"]{font-family:"DM Sans",sans-serif;}
.stApp{background:var(--oatmeal);color:var(--text);}
header[data-testid="stHeader"],#MainMenu,footer{display:none !important;}
.block-container{max-width:1200px;padding:0 4% 0 !important;}
.navbar{display:flex;justify-content:space-between;align-items:center;padding:25px 0;border-bottom:1px solid rgba(59,16,51,.12);}
.brand{font-size:13px;font-weight:700;letter-spacing:2px;color:var(--plum-dark);}
.brand-mark{font-size:20px;color:var(--plum-bright);margin-right:10px;}
.nav-tag{font-size:12px;letter-spacing:1px;color:var(--muted);text-transform:uppercase;}
.eyebrow{font-size:11px;font-weight:700;letter-spacing:2.5px;color:var(--plum-bright);margin-bottom:16px;}
.hero-text{padding-top:70px;}
.hero-text h1{font-family:"Playfair Display",serif;font-size:clamp(44px,6vw,78px);line-height:.98;letter-spacing:-2px;color:var(--plum-dark);}
.hero-text h1 span{display:block;color:var(--plum-bright);font-style:italic;}
.hero-description{max-width:520px;margin-top:26px;font-size:17px;line-height:1.8;color:var(--muted);}
div[data-testid="stVerticalBlockBorderWrapper"]{background:var(--cream);border:1px solid var(--border) !important;border-radius:24px !important;padding:30px 26px;box-shadow:var(--shadow);margin-top:40px;}
.upload-head{text-align:center;}
.upload-icon{width:64px;height:64px;margin:0 auto 20px;border-radius:50%;background:var(--plum-dark);color:#fff;display:flex;align-items:center;justify-content:center;font-size:28px;}
.upload-head h2{font-family:"Playfair Display",serif;font-size:30px;color:var(--plum-dark);margin-bottom:6px;}
.upload-head p{color:var(--muted);font-size:13px;margin-bottom:18px;}
[data-testid="stFileUploader"] section{background:var(--cream-dark);border:1px dashed var(--border);border-radius:14px;}
[data-testid="stFileUploader"] section *{color:var(--plum-dark) !important;}
[data-testid="stFileUploader"] button{background:var(--cream);border:1px solid var(--border);color:var(--plum-dark) !important;border-radius:10px;font-weight:700;}
[data-testid="stImage"] img{border-radius:14px;border:1px solid var(--border);}
.stButton>button{width:100%;background:var(--plum-dark);border:none;border-radius:13px;padding:.9rem;transition:.2s;}
.stButton>button p{color:#fff !important;font-weight:700;font-size:15px;}
.stButton>button:hover{background:var(--plum-bright);transform:translateY(-2px);}
.error-box{background:var(--berry-light);border:1px solid #D9B5C1;color:var(--berry);padding:16px 20px;border-radius:12px;font-size:14px;margin-top:20px;}
.result-section{padding:50px 0 20px;}
.result-section h2{font-family:"Playfair Display",serif;font-size:42px;color:var(--plum-dark);margin-bottom:26px;}
.result-layout{display:grid;grid-template-columns:.85fr 1.15fr;gap:25px;}
.uploaded-image-card{background:var(--cream);border:1px solid var(--border);border-radius:20px;padding:20px;box-shadow:var(--shadow);}
.card-label{font-size:10px;font-weight:700;letter-spacing:2px;color:var(--muted);margin-bottom:15px;}
.uploaded-image-card img{width:100%;height:370px;object-fit:contain;background:var(--cream-dark);border-radius:13px;}
.image-name{margin-top:12px;color:var(--muted);font-size:12px;word-break:break-all;}
.result-card{border-radius:20px;padding:45px;min-height:470px;display:flex;flex-direction:column;justify-content:center;text-align:center;border:1px solid var(--border);}
.result-real{background:var(--real-light);border-color:#B8D8C5;}
.result-tampered,.result-ai{background:var(--berry-light);border-color:#D9B5C1;}
.result-inconclusive{background:var(--cream-dark);}
.result-label{font-size:10px;letter-spacing:2.5px;font-weight:700;color:var(--muted);margin-bottom:25px;}
.result-icon{width:62px;height:62px;margin:0 auto 20px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:28px;font-weight:700;color:#fff;background:var(--plum);}
.result-real .result-icon{background:var(--real);}
.result-tampered .result-icon,.result-ai .result-icon{background:var(--berry);}
.result-title{font-family:"Playfair Display",serif;font-size:48px;line-height:1;color:var(--plum-dark);margin-bottom:20px;}
.result-comment{font-size:20px;font-weight:700;color:var(--text);margin-bottom:12px;}
.result-detail{max-width:600px;margin:0 auto 6px;font-size:14px;line-height:1.7;color:var(--muted);}
.confidence{margin-top:30px;text-align:left;}
.confidence-top{display:flex;justify-content:space-between;margin-bottom:9px;font-size:12px;color:var(--muted);}
.confidence-top strong{color:var(--plum-dark);font-size:13px;}
.confidence-bar{height:8px;background:rgba(59,16,51,.12);border-radius:20px;overflow:hidden;}
.confidence-fill{height:100%;background:var(--plum-bright);border-radius:20px;}
.technical-card{margin-top:25px;background:var(--cream);border:1px solid var(--border);border-radius:20px;padding:35px;box-shadow:var(--shadow);}
.technical-card h3{font-family:"Playfair Display",serif;font-size:28px;color:var(--plum-dark);margin-bottom:25px;}
.technical-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;}
.technical-item{display:flex;gap:18px;padding:20px;background:var(--cream-dark);border-radius:14px;}
.technical-number{font-size:12px;font-weight:700;color:var(--plum-bright);padding-top:3px;}
.technical-item strong{color:var(--plum-dark);font-size:15px;}
.technical-item p{margin-top:6px;font-size:13px;line-height:1.5;color:var(--muted);}
.technical-item small{display:block;margin-top:10px;font-size:11px;color:var(--plum-bright);font-weight:700;}
.how-section{padding:70px 0 60px;border-top:1px solid rgba(59,16,51,.12);margin-top:50px;}
.how-section h2{font-family:"Playfair Display",serif;font-size:44px;color:var(--plum-dark);margin-bottom:12px;}
.how-intro{color:var(--muted);line-height:1.7;font-size:15px;margin-bottom:40px;max-width:600px;}
.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:20px;}
.step{padding:25px;background:rgba(251,247,240,.55);border:1px solid var(--border);border-radius:16px;}
.step>span{color:var(--plum-bright);font-size:12px;font-weight:700;letter-spacing:1px;}
.step h3{margin:28px 0 9px;font-family:"Playfair Display",serif;font-size:23px;color:var(--plum-dark);}
.step p{font-size:13px;line-height:1.6;color:var(--muted);}
.disclaimer{max-width:1000px;margin:0 auto 60px;padding:22px 25px;border-left:3px solid var(--plum-bright);background:rgba(251,247,240,.5);}
.disclaimer strong{color:var(--plum-dark);font-size:13px;}
.disclaimer p{margin-top:6px;color:var(--muted);font-size:12px;line-height:1.6;}
.site-footer{padding:25px 0;border-top:1px solid rgba(59,16,51,.12);display:flex;justify-content:space-between;color:var(--muted);font-size:11px;letter-spacing:.5px;}
@media(max-width:900px){.result-layout,.technical-grid{grid-template-columns:1fr;}.steps{grid-template-columns:1fr 1fr;}}
@media(max-width:600px){.steps{grid-template-columns:1fr;}.nav-tag{display:none;}.result-card{padding:30px 20px;}.result-title{font-size:40px;}.site-footer{flex-direction:column;gap:8px;}}
"""


def squash(text):
    """Remove blank lines and indentation so Streamlit's markdown keeps the HTML intact."""
    return "".join(line.strip() for line in text.splitlines())


st.markdown("<style>" + "\n".join(l for l in CSS.splitlines() if l.strip()) + "</style>",
            unsafe_allow_html=True)

# ---------------------------------------------------------------- logic
COMMENTS = {
    "TAMPERED": ("Signs of manipulation detected",
                 "The ELA-based detector found compression inconsistencies that may indicate edited regions."),
    "AI-GENERATED": ("Likely AI-generated",
                     "The CIFAKE model found patterns commonly seen in synthetic images."),
    "REAL": ("No strong signs of manipulation",
             "The deciding detector did not flag this image. This is a model output, not proof of authenticity."),
    "INCONCLUSIVE": ("The models are not confident",
                     "Both detectors scored near the middle, so no reliable verdict can be given for this image."),
}
ICONS = {"REAL": "✓", "TAMPERED": "!", "AI-GENERATED": "✦", "INCONCLUSIVE": "?"}
CLASSES = {"REAL": "result-real", "TAMPERED": "result-tampered",
           "AI-GENERATED": "result-ai", "INCONCLUSIVE": "result-inconclusive"}


def casia_label(score):
    # CASIA: high score = TAMPERED
    if score >= HIGH:
        return "TAMPERED"
    return "REAL" if score <= LOW else "INCONCLUSIVE"


def cifake_label(score):
    # CIFAKE (project convention): high score = REAL, low score = AI-GENERATED
    if score >= HIGH:
        return "REAL"
    return "AI-GENERATED" if score <= LOW else "INCONCLUSIVE"


def decide(casia_score, cifake_score):
    """A detector that flags something (TAMPERED / AI-GENERATED) outranks a REAL.
    Within each group, the detector further from 0.5 (more confident) decides."""
    options = []
    for name, label, score in (("CASIA + ELA", casia_label(casia_score), casia_score),
                               ("CIFAKE CNN", cifake_label(cifake_score), cifake_score)):
        if label != "INCONCLUSIVE":
            options.append((label != "REAL", abs(score - 0.5), name, label, score))
    if not options:
        return "INCONCLUSIVE", None, None
    _, _, name, label, score = max(options)
    return label, name, max(score, 1 - score)


def to_data_uri(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img.thumbnail((900, 900))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def result_html(file_name, data_uri, casia_score, cifake_score):
    verdict, decided_by, confidence = decide(casia_score, cifake_score)
    comment, detail = COMMENTS[verdict]
    by = (f"Decided by: {decided_by}" if decided_by
          else "Neither detector was confident enough to decide.")
    bar = ""
    if confidence is not None:
        bar = (f'<div class="confidence"><div class="confidence-top"><span>Model confidence</span>'
               f'<strong>{confidence * 100:.1f}%</strong></div><div class="confidence-bar">'
               f'<div class="confidence-fill" style="width:{confidence * 100:.1f}%"></div></div></div>')
    return squash(f"""
    <section class="result-section">
      <p class="eyebrow">ANALYSIS COMPLETE</p>
      <h2>Detection Result</h2>
      <div class="result-layout">
        <div class="uploaded-image-card">
          <div class="card-label">UPLOADED IMAGE</div>
          <img src="{data_uri}" alt="Uploaded image">
          <p class="image-name">{html.escape(file_name)}</p>
        </div>
        <div class="result-card {CLASSES[verdict]}">
          <div class="result-label">FINAL CLASSIFICATION</div>
          <div class="result-icon">{ICONS[verdict]}</div>
          <h3 class="result-title">{verdict}</h3>
          <p class="result-comment">{comment}</p>
          <p class="result-detail">{detail}</p>
          <p class="result-detail">{by}</p>
          {bar}
        </div>
      </div>
      <div class="technical-card">
        <p class="eyebrow">TECHNICAL ANALYSIS</p>
        <h3>Models Used</h3>
        <div class="technical-grid">
          <div class="technical-item"><span class="technical-number">01</span><div>
            <strong>CASIA + ELA</strong>
            <p>Detects signs of traditional image manipulation and tampering.</p>
            <small>Score: {casia_score:.4f} · {casia_label(casia_score)}</small></div></div>
          <div class="technical-item"><span class="technical-number">02</span><div>
            <strong>CIFAKE CNN</strong>
            <p>Detects patterns associated with AI-generated images.</p>
            <small>Score: {cifake_score:.4f} · {cifake_label(cifake_score)}</small></div></div>
        </div>
      </div>
    </section>
    """)


# ---------------------------------------------------------------- page
st.markdown(squash("""
<div class="navbar"><div class="brand"><span class="brand-mark">◈</span>IMAGE INTEGRITY</div>
<div class="nav-tag">Deep Learning Analysis</div></div>
"""), unsafe_allow_html=True)

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown(squash("""
    <div class="hero-text"><p class="eyebrow">IMAGE AUTHENTICITY ANALYZER</p>
    <h1>Is this image <span>authentic?</span></h1>
    <p class="hero-description">Upload an image and let our deep-learning models analyze it
    for signs of manipulation or AI-generated content.</p></div>
    """), unsafe_allow_html=True)

with right:
    with st.container(border=True):
        st.markdown(squash("""
        <div class="upload-head"><div class="upload-icon">↑</div><h2>Analyze an image</h2>
        <p>JPG, JPEG or PNG · Maximum 10 MB</p></div>
        """), unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"],
                                    label_visibility="collapsed")
        if uploaded is not None:
            st.image(uploaded, use_container_width=True)
        analyze = st.button("Analyze Image")

if analyze:
    if uploaded is None:
        st.markdown('<div class="error-box">Please select an image.</div>', unsafe_allow_html=True)
    elif uploaded.size > MAX_MB * 1024 * 1024:
        st.markdown(f'<div class="error-box">File is larger than {MAX_MB} MB.</div>',
                    unsafe_allow_html=True)
    else:
        suffix = os.path.splitext(uploaded.name)[1].lower() or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            path = tmp.name
        try:
            with st.spinner("Analyzing image..."):
                _, casia_score = predict_casia(path)
                _, cifake_score = predict_cifake(path)
            st.markdown(result_html(uploaded.name, to_data_uri(uploaded.getvalue()),
                                    casia_score, cifake_score), unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="error-box">Error while analyzing image: {html.escape(str(e))}</div>',
                        unsafe_allow_html=True)
        finally:
            os.remove(path)

st.markdown(squash("""
<section class="how-section">
  <p class="eyebrow">THE PROCESS</p><h2>How it works</h2>
  <p class="how-intro">Two deep-learning approaches analyze different types of image authenticity.</p>
  <div class="steps">
    <div class="step"><span>01</span><h3>Upload</h3><p>Select an image from your device.</p></div>
    <div class="step"><span>02</span><h3>Preprocess</h3><p>The image is prepared for the appropriate neural network.</p></div>
    <div class="step"><span>03</span><h3>Analyze</h3><p>CASIA with ELA checks for tampering, while CIFAKE checks for AI generation.</p></div>
    <div class="step"><span>04</span><h3>Classify</h3><p>The system provides one final classification, decided by the more confident detector.</p></div>
  </div>
</section>
<section class="disclaimer"><strong>Important</strong>
<p>Results are model-based predictions and should not be treated as definitive proof of image
authenticity. Performance may vary depending on the image.</p></section>
<div class="site-footer"><span>Image Integrity Analyzer</span><span>Deep Learning · CNN · ELA</span></div>
"""), unsafe_allow_html=True)
