import streamlit as st
from transformers import pipeline
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="AI Vision Classifier", page_icon="👁️")
st.title("👁️ AI Vision Classifier")
st.write("Upload an image and the AI will identify the primary subject using a Vision Transformer.")

# 2. Load the Vision Model (Cached so it doesn't download on every click)
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

with st.spinner("Loading AI Model... (This takes a moment on the first run)"):
    classifier = load_model()

# 3. Build the UI
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Run inference and display results
    st.subheader("AI Predictions:")
    with st.spinner("Analyzing pixels..."):
        results = classifier(image)
        
        for res in results[:3]:
            label = res['label'].capitalize()
            confidence = round(res['score'] * 100, 1)
            st.write(f"• **{label}**: {confidence}%")
