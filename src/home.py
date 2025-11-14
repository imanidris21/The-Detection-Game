# Home.py
# streamlit run src/home.py


import streamlit as st
from backend.utils import init_db, load_images_meta
import pkgutil, sys


print("HAS_SQLALCHEMY:", pkgutil.find_loader("sqlalchemy") is not None, sys.version)


st.set_page_config(page_title="The detection Game — Home", layout="wide", initial_sidebar_state="collapsed")


# Initialize database and load data
init_db()

# Main content: title and subheading
st.markdown("<h1 style='text-align: center;'>Human or Machine?</h1>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; max-width: 800px; margin: 0 auto; padding: 0 2rem;'>
<h3>Do you think you can spot the AI-generated artworks and beat our AI Detector</h3>

</div>
""", unsafe_allow_html=True)

# Simple styling
# st.markdown("""
# <style>
# /* Button styling */
# .stButton {
#     display: flex !important;
#     justify-content: center !important;
#     margin-top: 2rem !important;
# }
# .stButton > button {
#     background-color: #0066cc !important;
#     color: white !important;
#     border: none !important;
#     padding: 0.5rem 1.5rem !important;
#     border-radius: 8px !important;
#     font-weight: 600 !important;
#     font-size: 1rem !important;
# }
# .stButton > button:hover {
#     background-color: #0052a3 !important;
# }
# </style>
# """, unsafe_allow_html=True)

# col1, col2, col3 = st.columns([2, 1, 2])
# with col2:
#     if st.button("Start the Game", type="primary"):
#         st.switch_page("pages/1_Take_the_Test.py")



center_col = st.columns([1, 1, 1])[1]

with center_col:
    start = st.button("Start the Game", type="primary")

if start:
    st.switch_page("pages/1_Take_the_Test.py")