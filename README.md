# LangSAM and Stable Diffusion Inpainting

This project showcases a method for object detection, segmentation, and inpainting using the LangSAM library and Stable Diffusion model. The LangSAM library combines GroundingDino and SAM for effective object segmentation, while the Stable Diffusion InpaintPipeline is used for inpainting the segmented areas based on target prompts. The project involves installing the necessary dependencies, initializing the models, and running a final pipeline that processes an image to perform segmentation and inpainting, transforming objects in the image based on user-defined prompts.

# Installation
pip install -r requirements.txt

# Local Deployment using streamlit
streamlit run streamlit_app.py
