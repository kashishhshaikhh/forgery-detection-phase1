import os
import tempfile

import streamlit as st
from PIL import Image

# These import the models once per server process (not on every click).
from casia_predict import predict_casia
from cifake_predict import predict_cifake

# Display bands. Scores between LOW and HIGH are shown as INCONCLUSIVE.
# This is a presentation choice, not a validated forensic threshold.
LOW = 0.30
HIGH = 0.70


def casia_label(score):
    # CASIA: high score = TAMPERED
    if score >= HIGH:
        return "TAMPERED"
    if score <= LOW:
        return "REAL"
    return "INCONCLUSIVE"


def cifake_label(score):
    # Verified on CIFAKE test set: high score = AI-GENERATED, low = REAL
    if score >= HIGH:
        return "AI-GENERATED"
    if score <= LOW:
        return "REAL"
    return "INCONCLUSIVE"


st.set_page_config(page_title="Image Forgery & Synthesis Detection", layout="centered")

st.title("Image Forgery and Synthesis Detection System")
st.caption("Phase 1: traditional tampering (CASIA + ELA + CNN) and AI-generated image detection (CIFAKE CNN)")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    st.image(uploaded, caption="Uploaded image", use_container_width=True)

    if st.button("Analyze image", type="primary"):
        suffix = os.path.splitext(uploaded.name)[1].lower() or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            path = tmp.name

        try:
            with st.spinner("Analyzing..."):
                _, casia_score = predict_casia(path)
                _, cifake_score = predict_cifake(path)
        except Exception as e:
            st.error(f"Error while analyzing image: {e}")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Traditional Tampering")
                st.caption("CASIA + ELA + CNN")
                st.metric("Result", casia_label(casia_score))
                st.write(f"Score: {casia_score:.4f}")

            with col2:
                st.subheader("AI-Generated Detection")
                st.caption("CIFAKE CNN")
                st.metric("Result", cifake_label(cifake_score))
                st.write(f"Score: {cifake_score:.4f}")

            st.info(
                "These are model outputs, not proof of authenticity. "
                "INCONCLUSIVE means the model was not confident."
            )
        finally:
            os.remove(path)

with st.expander("Methodology and limitations"):
    st.write(
        "**CASIA + ELA + CNN:** Error Level Analysis re-saves the image as a JPEG and "
        "highlights differences, which a CNN trained on CASIA v2.0 classifies. "
        "**CIFAKE CNN:** the image is resized to 32x32 and classified as real or "
        "AI-generated. Both models were trained on limited datasets and may not "
        "generalize to all real-world photos. Deepfake detection is planned for Phase 2."
    )
