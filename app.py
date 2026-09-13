import gradio as gr
from transformers import pipeline

# 1. Load the pre-trained Vision model (Downloads on first run)
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

# 2. Define the prediction function
def classify_image(image):
    if image is None:
        return "Please upload an image."
    
    # Run inference on the image
    results = classifier(image)
    
    # Format the top 3 predictions cleanly
    output = ""
    for res in results[:3]:
        label = res['label'].capitalize()
        confidence = round(res['score'] * 100, 1)
        output += f"• {label}: {confidence}%\n"
        
    return output

# 3. Build the User Interface
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload any photo"),
    outputs=gr.Text(label="AI Predictions"),
    title="👁️ AI Vision Classifier",
    description="Upload an image and the AI will identify the primary subject using a Vision Transformer."
    # Removed 'theme' to fix the version warning
)

# 4. Launch the app and force browser to open
if __name__ == "__main__":
    demo.launch(inbrowser=True)