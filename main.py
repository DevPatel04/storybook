# Install necessary packages
import torch
import os
from diffusers import StableDiffusionPipeline
from huggingface_hub import login
import streamlit as st

# Authenticate with Hugging Face
HUGGINGFACE_TOKEN = st.secrets['HF_TOKEN'] # Replace with your token
login(token=HUGGINGFACE_TOKEN)

# Load the Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

# Define training parameters
INSTANCE_PROMPT = "photo of sks person"  # Change 'sks' to a unique keyword
CLASS_PROMPT = "photo of a person"
UPLOAD_DIR = "/kaggle/input/virat-kohli/Virat Kohli Facial Recognition/known_faces/Virat_Kohli"  # Update with your actual path
MODEL_SAVE_PATH = "./trained_dreambooth_model"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Clone the DreamBooth training script from Hugging Face
!git clone https://github.com/huggingface/diffusers.git
!cd diffusers/examples/dreambooth && pip install -r requirements.txt

# Run Training
!accelerate launch diffusers/examples/dreambooth/train_dreambooth.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --instance_data_dir="{UPLOAD_DIR}" \
    --output_dir="{MODEL_SAVE_PATH}" \
    --instance_prompt="{INSTANCE_PROMPT}" \
    --class_prompt="{CLASS_PROMPT}" \
    --num_class_images=200 \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=400 

# Save the trained model
pipeline.save_pretrained(MODEL_SAVE_PATH)
print("✅ Model training complete! Saved at:", MODEL_SAVE_PATH)
