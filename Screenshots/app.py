
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide"
)

# ── Load model, class map, normalization params, config ───────
@st.cache_resource
def load_resources():
    """
    @st.cache_resource: Loads everything once and caches it.
    Without this, the 287K-parameter model reloads on every
    image upload — causing a 5-10 second delay each time.
    """
    model        = tf.keras.models.load_model("cnn_final.keras")
    # Standard cross-entropy model loads without custom_objects

    with open("class_indices.json") as f:
        idx_to_class = json.load(f)
    # {'0':'Glioma', '1':'Meningioma', '2':'No Tumor', '3':'Pituitary'}

    with open("norm_params.json") as f:
        norm = json.load(f)
    global_mean = np.array(norm["mean"], dtype=np.float32)
    global_std  = np.array(norm["std"],  dtype=np.float32)
    # These MUST match what was used during training

    with open("config.json") as f:
        config = json.load(f)

    return model, idx_to_class, global_mean, global_std, config

model, idx_to_class, GLOBAL_MEAN, GLOBAL_STD, config = load_resources()

NUM_CLASSES       = len(idx_to_class)
NOTUMOR_IDX       = config["notumor_idx"]
NOTUMOR_THRESHOLD = config["notumor_threshold"]

# ── Preprocessing: must EXACTLY match training ─────────────────
def preprocess_image(pil_img):
    """
    Apply the same normalization used during training.
    If this doesn't match, the model receives out-of-distribution
    input and predictions will be wrong — most common deployment bug.
    """
    img = pil_img.resize((224, 224))
    # PIL resize uses bilinear interpolation by default

    arr = np.array(img, dtype=np.float32) / 255.0
    # Scale to [0,1] first — same as training

    arr = (arr - GLOBAL_MEAN) / (GLOBAL_STD + 1e-8)
    # Z-score normalise using the same stats computed from the dataset

    return np.expand_dims(arr, axis=0)
    # (224,224,3) → (1,224,224,3): add batch dimension

# ── Threshold-based prediction ─────────────────────────────────
def predict_with_threshold(probs):
    """
    Medical-safe prediction: only return 'No Tumor' when
    the model is >= NOTUMOR_THRESHOLD confident.
    Below that, predict the most likely tumor class instead.
    This minimises false negatives (missed tumors).
    """
    if probs[NOTUMOR_IDX] >= NOTUMOR_THRESHOLD:
        idx = int(NOTUMOR_IDX)
    else:
        masked          = probs.copy()
        masked[NOTUMOR_IDX] = 0.0
        idx = int(np.argmax(masked))
    return idx, float(probs[idx]) * 100

# ── Clinical information per class ────────────────────────────
TUMOR_INFO = {
    "Glioma": {
        "desc"    : "Glioma originates in the glial cells of the brain or spine. "
                    "It is the most common type of malignant primary brain tumor.",
        "severity": "🔴 High Risk",
        "action"  : "Immediate referral to a neurosurgeon or neuro-oncologist is strongly advised.",
        "color"   : "#C62828"
    },
    "Meningioma": {
        "desc"    : "Meningioma arises from the meninges — the membranes surrounding "
                    "the brain and spinal cord. Usually benign and slow-growing.",
        "severity": "🟠 Moderate Risk",
        "action"  : "Schedule neurological evaluation and follow-up MRI imaging.",
        "color"   : "#E65100"
    },
    "Pituitary": {
        "desc"    : "Pituitary tumors develop in the pituitary gland at the base of "
                    "the brain, often affecting hormone regulation.",
        "severity": "🟡 Moderate Risk",
        "action"  : "Consult an endocrinologist and neurosurgeon for full assessment.",
        "color"   : "#F9A825"
    },
    "No Tumor": {
        "desc"    : "No tumor detected in the MRI scan.",
        "severity": "🟢 No Risk Detected",
        "action"  : "No immediate action required. Maintain regular health check-ups.",
        "color"   : "#2E7D32"
    }
}

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    **Brain Tumor MRI Classifier**

    Classifies brain MRI scans into:
    - Glioma
    - Meningioma
    - Pituitary Tumor
    - No Tumor

    **Model:** Custom CNN (3 conv blocks)
    **Dataset:** Alam Shihab (Kaggle)
    **Optimised for:** F2 score (recall-weighted)
    """)
    st.warning("⚠️ Research tool only.\nNot a substitute for clinical diagnosis.")
    st.markdown(f"**No Tumor threshold:** `{NOTUMOR_THRESHOLD}`")
    st.markdown(f"**Model F2 score:** `{config.get('best_weighted_f2', 'N/A')}%`")

# ── Main page ─────────────────────────────────────────────────
st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("*Upload a brain MRI scan — the model classifies the tumor type*")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"],
    help="Upload a T1 or T2 weighted axial brain MRI scan"
)

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    # .convert("RGB"): force 3 channels — handles grayscale MRI uploads

    with st.spinner("Analysing MRI..."):
        img_batch  = preprocess_image(pil_img)
        raw_probs  = model.predict(img_batch, verbose=0)[0]
        pred_idx, confidence = predict_with_threshold(raw_probs)
        pred_class = idx_to_class[str(pred_idx)]

    # ── Layout ────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 Uploaded MRI Scan")
        st.image(pil_img, use_column_width=True)

    with col2:
        st.subheader("📊 Class Probabilities")
        st.caption("(Raw model outputs before threshold adjustment)")
        for i in range(NUM_CLASSES):
            cls_name = idx_to_class[str(i)]
            prob     = float(raw_probs[i]) * 100
            is_pred  = (i == pred_idx)

            label = f"**{cls_name} ← Predicted**" if is_pred else cls_name
            st.markdown(label)
            st.progress(prob / 100)
            st.caption(f"{prob:.2f}%")

    st.markdown("---")
    st.subheader("🩺 Diagnosis Report")

    info     = TUMOR_INFO.get(pred_class, TUMOR_INFO["No Tumor"])
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown(f"""
        <div style="background:{info['color']}22;
                    border-left:6px solid {info['color']};
                    padding:20px; border-radius:8px;">
            <h2 style="color:{info['color']};margin:0">{pred_class.upper()}</h2>
            <h4 style="margin:8px 0">{info['severity']}</h4>
            <p style="font-size:18px;margin:0">
                Confidence: <b>{confidence:.2f}%</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"**About:** {info['desc']}")
        st.markdown(f"**Recommendation:** {info['action']}")
        if pred_class == "No Tumor":
            st.info(
                f"ℹ️ 'No Tumor' only predicted when model confidence "
                f"≥ {NOTUMOR_THRESHOLD*100:.0f}%. "
                f"This threshold minimises missed tumors."
            )

    st.markdown("---")
    st.caption(
        "⚠️ For educational and research purposes only. "
        "Always consult a qualified medical professional for diagnosis."
    )
