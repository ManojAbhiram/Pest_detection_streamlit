import streamlit as st

# ✅ MUST BE FIRST Streamlit command
st.set_page_config(page_title="Pest Detection", page_icon="🪲", layout="centered")

import numpy as np
from PIL import Image
import os
import gdown
from tensorflow.keras.models import load_model

# -------------------- MODEL SETUP ----------------------

MODEL_PATH = "finalmodel3.h5"
MODEL_URL = "https://drive.google.com/uc?id=1eZobQPU-lgm7_wLyAQQdlO6uQzU3xeSv"

@st.cache_resource(show_spinner=False)
def get_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading large model file... Please wait (~2GB)"):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            st.success("✅ Model downloaded successfully!")
    return load_model(MODEL_PATH)

model = get_model()

# -------------------- MAPPINGS ----------------------

pest_names = [
    "Termites",
    "Thrips",
    "Tussock caterpillar",
    "Shoot Borer",
]

pest_info = [
    {
        "Cultural Method": "There is no specific cultural method for managing Termites.",
        "Chemical Method": "Soil Treatment: Apply termiticides to the soil around a building's foundation.",
        "Wood Treatment": "Apply certain chemicals directly to wooden structures.",
        "Biological Method": "There is no specific biological method for managing termites.",
    },
    {
        "Cultural Method": "Submerge infected crops intermittently for 1-2 days.\nDrag a wet cloth on seedlings.\nFlood for 2 days.\nUse resistant cultivars.",
        "Chemical Method": "Apply Phosphamidon 40 SL when thrips population exceeds threshold.",
        "Wood Treatment": "",
        "Biological Method": "Predatory thrips, coccinellid beetles, anthocorid bugs, and staphylinid beetles feed on thrips.",
    },
    {
        "Cultural Method": "There is no specific cultural method for managing Tussock Caterpillars.",
        "Chemical Method": "Use Supreme IT Insecticide to treat yard and ornamentals.",
        "Wood Treatment": "",
        "Biological Method": "No specific biological method mentioned.",
    },
    {
        "Cultural Method": "Remove affected terminal shoots and fruits.\nAvoid continuous cropping of brinjal.\nGrow resistant varieties.\nEncourage parasitoids.",
        "Chemical Method": "Use Neem Seed Kernel Extract (NSKE) or other recommended chemicals at intervals.",
        "Wood Treatment": "",
        "Biological Method": "No specific biological method for managing shoot borers.",
    },
]

# -------------------- STREAMLIT UI ----------------------

st.title("🪲 Pest Detection and Prevention System")

uploaded_file = st.file_uploader("📤 Upload an image of the pest", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    resized_image = image.resize((256, 256))
    np_image = np.array(resized_image) / 255.0
    prediction = model.predict(np.expand_dims(np_image, axis=0))[0]
    pred_index = int(np.argmax(prediction))
    pred_name = pest_names[pred_index]

    st.success(f"✅ **Predicted Pest:** {pred_name}")

    st.subheader("🧪 Prevention & Treatment Methods")
    pest_data = pest_info[pred_index]
    for method, advice in pest_data.items():
        if advice.strip():
            st.markdown(f"**{method}:**\n{advice}")
