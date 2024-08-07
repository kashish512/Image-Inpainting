# -*- coding: utf-8 -*-
"""streamlit_app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1d7cwbkOcjvnZ5B51nSQZeGuNpMH-zTTR
"""

import streamlit as st
from PIL import Image
import io
from app import final_pipeline  # Import the function from app.py

st.title("Image Inpainting with LangSAM and Stable Diffusion")
st.write("Upload an image, and specify source and target prompts for inpainting.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
source_prompt = st.text_input("Source Prompt", value="House")
target_prompt = st.text_input("Target Prompt", value="Middle-age Castle")

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB").resize((1064, 1064))
    st.image(image_pil, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Processing...")

    try:
        result_image = final_pipeline(image_pil, source_prompt, target_prompt)
        st.image(result_image, caption='Inpainted Image', use_column_width=True)
    except ValueError as e:
        st.error(e)