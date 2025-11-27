# app.py
import streamlit as st
from PIL import Image
import numpy as np
import io
import traceback

# Keras models and preprocessors
from tensorflow.keras.applications import (
    ResNet50, ResNet50V2, Xception, InceptionV3, MobileNetV2,
    DenseNet121, NASNetMobile, NASNetLarge, EfficientNetV2B0
)
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnetv2_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficient_preprocess

# Generic decode util for ImageNet (works across Keras application models)
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# ---------------------------
# App styling (eye-catching)
# ---------------------------
PAGE_STYLE = """
<style>
/* Full-screen gradient background */
[data-testid="stAppViewContainer"]{
  background: radial-gradient(circle at 10% 10%, rgba(255,255,255,0.04), transparent 15%),
              linear-gradient(120deg, #0f172a, #0f4c81 25%, #5b21b6 60%, #ec4899 100%);
  background-attachment: fixed;
}

/* Card */
.card {
  background: rgba(255,255,255,0.06);
  border-radius: 16px;
  padding: 20px;
  box-shadow: 0 8px 30px rgba(2,6,23,0.6);
  backdrop-filter: blur(6px);
  border: 1px solid rgba(255,255,255,0.04);
  color: #ffffff;
}

/* Header */
h1 { font-weight: 700 !important; color: #fff; }

/* Small labels */
.small {
  color: rgba(255,255,255,0.8);
  font-size: 14px;
}
</style>
"""

st.markdown(PAGE_STYLE, unsafe_allow_html=True)

# ---------------------------
# Model registry & metadata
# ---------------------------
MODEL_OPTIONS = {
    "ResNet50": (ResNet50, resnet_preprocess, (224, 224)),
    "ResNet50V2": (ResNet50V2, resnetv2_preprocess, (224, 224)),
    "Xception": (Xception, xception_preprocess, (299, 299)),
    "InceptionV3": (InceptionV3, inception_preprocess, (299, 299)),
    "MobileNetV2": (MobileNetV2, mobilenet_preprocess, (224, 224)),
    "DenseNet121": (DenseNet121, densenet_preprocess, (224, 224)),
    "NASNetMobile": (NASNetMobile, nasnet_preprocess, (224, 224)),
    "NASNetLarge": (NASNetLarge, nasnet_preprocess, (331, 331)),
    "EfficientNetV2B0": (EfficientNetV2B0, efficient_preprocess, (224, 224)),
}

# ---------------------------
# Caching models
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    """Load and return the selected pretrained Keras model (ImageNet weights)."""
    model_cls, _, _ = MODEL_OPTIONS[model_name]
    model = model_cls(weights="imagenet")
    return model

# ---------------------------
# UI Layout
# ---------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.title("📸 Keras Multi-Model Classifier")
st.markdown("<div class='small'>Upload an image, pick a pretrained Keras model, and get ImageNet predictions.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload image (jpg, png)", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    model_name = st.selectbox("Choose model", list(MODEL_OPTIONS.keys()))
    top_k = st.slider("Top K predictions", min_value=1, max_value=10, value=3)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br />", unsafe_allow_html=True)

# ---------------------------
# Prediction area
# ---------------------------
if uploaded is not None:
    try:
        # Read image
        image_bytes = uploaded.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Show original
        st.image(img, caption="Uploaded image", use_column_width=True)

        # Load model lazily
        with st.spinner(f"Loading {model_name} (this happens only once per session)..."):
            model = load_model(model_name)

        # Preprocess for the chosen model
        _, preprocess_fn, target_size = MODEL_OPTIONS[model_name]

        # Resize to required input size
        img_resized = img.resize(target_size)
        arr = np.array(img_resized).astype("float32")
        arr = np.expand_dims(arr, axis=0)

        # Some preprocessors expect channels_last in specific ranges
        x = preprocess_fn(arr)

        # Predict button
        if st.button("🔍 Predict"):
            with st.spinner("Running inference..."):
                preds = model.predict(x)
                decoded = decode_predictions(preds, top=top_k)[0]

            # Results card
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("🔮 Predictions")
            for rank, (_id, label, prob) in enumerate(decoded, start=1):
                st.write(f"**{rank}. {label}** — {prob*100:.2f}%")
            top_index = int(np.argmax(preds[0]))
            st.success(f"🏆 Top class index: {top_index}")
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error("Something went wrong while processing the image or running the model.")
        st.text(traceback.format_exc())
else:
    st.info("Upload an image to get started.")

