from pathlib import Path
import tqdm
import torch
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
import cv2
import gradio as gr

# Configuration class
class CFG:
    device = "cuda"
    seed = 42
    generator = torch.manual_seed(seed)  # Fix here
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12

# Load the Stable Diffusion model
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    revision="fp16", use_auth_token='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', guidance_scale=CFG.image_gen_guidance_scale
)
image_gen_model = image_gen_model.to(CFG.device)

# Function to generate image from a prompt
def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=torch.Generator().manual_seed(CFG.seed),  # Use CPU generator
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image

# Gradio interface function
def gradio_generate_image(prompt):
    image = generate_image(prompt, image_gen_model)
    return image

# Create a Gradio interface
interface = gr.Interface(
    fn=gradio_generate_image, 
    inputs="text", 
    outputs="image",
    title="AI Image Generator",
    description="Generate images based on a text prompt using Stable Diffusion.",
)

# Launch the interface
if __name__ == "__main__":
    interface.launch()
