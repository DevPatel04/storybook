import streamlit as st
import torch
import os
import shutil
from diffusers import StableDiffusionPipeline
from PIL import Image

def train_dreambooth(instance_prompt, upload_dir, model_save_path):
    """Train DreamBooth model dynamically."""
    MODEL_ID = "runwayml/stable-diffusion-v1-5"
    os.makedirs(model_save_path, exist_ok=True)
    
    # Run training (assuming training script is available)
    os.system(f"accelerate launch diffusers/examples/dreambooth/train_dreambooth.py \
        --pretrained_model_name_or_path={MODEL_ID} \
        --instance_data_dir={upload_dir} \
        --output_dir={model_save_path} \
        --instance_prompt='{instance_prompt}' \
        --resolution=512 \
        --train_batch_size=1 \
        --gradient_accumulation_steps=1 \
        --learning_rate=5e-6 \
        --max_train_steps=400")
    return model_save_path

def generate_image(model_path, prompt):
    """Generate an image using the trained DreamBooth model."""
    pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    image = pipeline(prompt).images[0]
    return image

st.title("🖼️ AI-Powered Storybook Generator")
st.sidebar.header("Upload Character Images")

# Upload images
upload_dir = "uploaded_images"
shutil.rmtree(upload_dir, ignore_errors=True)
os.makedirs(upload_dir, exist_ok=True)
uploaded_files = st.sidebar.file_uploader("Upload character images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files:
    for uploaded_file in uploaded_files:
        with open(os.path.join(upload_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.sidebar.success("✅ Images uploaded successfully!")

# Train the model
if st.sidebar.button("Train Model"):
    instance_prompt = "photo of sks person"
    model_save_path = "trained_dreambooth_model"
    st.sidebar.text("Training model... (this may take some time)")
    train_dreambooth(instance_prompt, upload_dir, model_save_path)
    st.sidebar.success("✅ Model trained successfully!")

# Generate image
st.header("📜 Generate Story Scene")
prompt = st.text_input("Enter a scene description:", "photo of sks person in a forest")
if st.button("Generate Image"):
    model_path = "trained_dreambooth_model"
    st.text("Generating image...")
    generated_image = generate_image(model_path, prompt)
    st.image(generated_image, caption="Generated Scene", use_column_width=True)
    st.success("✅ Image generated successfully!")
    generated_image.save("generated_scene.png")
    st.download_button("Download Image", "generated_scene.png", "image/png")
