# Home.py
# streamlit run src/home.py


import streamlit as st
from backend.utils import init_db, load_images_meta
import pkgutil, sys
import base64

# Page configuration
st.set_page_config(page_title="The detection Game — Home", layout="wide", initial_sidebar_state="collapsed")

# Function to convert image to base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Initialize database and load data
init_db()

# Main content: title and subheading
st.markdown("<h1 style='text-align: center; font-size: 5rem; margin-top: 4rem;'>Human or Machine?</h1>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; max-width: 1000px; margin: 0 auto; padding: 0 1rem;'>
<h3 style='font-size: 2.5rem;'>Do you think you can spot the AI-generated artworks and beat our AI Detector</h3>

</div>
""", unsafe_allow_html=True)

# Get base64 image
bg_image = get_base64_of_image("src/assets/home_bg.jpg")

# Simple styling
st.markdown(f"""
<style>

/* Background image */
.stApp {{
    background-image: url("data:image/jpeg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

/* Force headings to be centered on all viewports */
div.block-container h1,
div.block-container h3 {{
    text-align: center !important;
}}



/* Button styling */
.stButton {{
    display: flex !important;
    justify-content: center !important;
    margin-top: 2rem !important;
}}
.stButton > button {{
    background-color: white !important;
    color: #262730 !important;
    border: 1px solid #d3d3d3 !important;
    padding: 0.5rem 1.5rem !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}}
.stButton > button:hover {{
    background-color: #ff4b4b !important;
    color: white !important;
    border: 1px solid #ff4b4b !important;
}}

/* Center align text in columns */
.stColumn {{
    text-align: center !important;
}}   

</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([3, 1, 3])
with col2:
    if st.button("Start the Game", type="primary",use_container_width=True):
        st.switch_page("pages/1_Take_the_Test.py")




