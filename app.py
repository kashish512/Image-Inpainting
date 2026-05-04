import streamlit as st
import torch
import numpy as np
from PIL import Image
from lang_sam import LangSAM
from diffusers import StableDiffusionInpaintPipeline
import os
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_models():
    sam_model = LangSAM()
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "sd-legacy/stable-diffusion-inpainting",
        torch_dtype=torch.float16)
    pipe.to(device)
    return sam_model, pipe

# UI Setup
st.title("Prompt based Inpainting Pipeline")
uploaded_file = st.file_uploader("Image", type=["jpg", "png", "jpeg"])
src_text = st.text_input("Object", "sunglasses")
tgt_text = st.text_input("Replace with", "goggles")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB").resize((512, 512))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Image")
        st.image(img, use_container_width=True)

    if st.button("Run Logic"):
        sam, pipe = load_models()

        # Step 1: LangSAM predict
        results = sam.predict([img], [src_text])
        result = results[0]

        masks = result["masks"]
        boxes = result["boxes"]
        labels = result["labels"]
        logits = result["scores"]
        mask_scores = result['mask_scores']

        # Step 2: Select best mask based on LangSAM masking
        best_idx = np.argmax(mask_scores)
        mask = masks[best_idx]

        # Step 3: Convert mask to PIL Image
        mask = (mask > 0.5).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask)

        # Step 4: Perform inpainting with StableDiffusionInpaintPipeline
        inpainted_image = pipe(prompt=tgt_text, image=[img], mask_image=mask_pil).images[0]

        with col2:
                st.subheader("Result")
                st.image(inpainted_image, use_container_width=True)
