# app.py
import streamlit as st
import pandas as pd
import numpy as np
from workingmodel import get_recommendations  # Import your existing functions

# UI Setup
st.set_page_config(page_title="Tamil Movie Recommender", page_icon="🎬")
st.title("🎬 Tamil Movie Recommender")

# Input
user_input = st.text_input("Enter a movie name or keywords (e.g., 'ரெமோ', 'காதல் நகைச்சுவை'):")

# Get Recommendations
if user_input:
    with st.spinner("Finding similar movies..."):
        recommendations = get_recommendations(user_input)
    
    if isinstance(recommendations, str):  # Error case
        st.warning(recommendations)
    else:
        st.success("Top Recommendations:")
        for movie in recommendations:
            st.write(f"- {movie}")