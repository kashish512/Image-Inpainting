# LangSAM and Stable Diffusion Inpainting

This project showcases a method for object detection, segmentation, and inpainting using the LangSAM library and Stable Diffusion model. The LangSAM library combines GroundingDino and SAM for effective object segmentation, while the Stable Diffusion InpaintPipeline is used for inpainting the segmented areas based on target prompts. The project involves installing the necessary dependencies, initializing the models, and running a final pipeline that processes an image to perform segmentation and inpainting, transforming objects in the image based on user-defined prompts.

# Installation
pip install -r requirements.txt

# Local hosting using streamlit
streamlit run app.py

<img width="902" height="847" alt="Screenshot 2026-05-05 002404" src="https://github.com/user-attachments/assets/d150a47e-797f-40e3-b378-68171898b00f" />

