import os
import streamlit as st

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'processed_hashes' not in st.session_state:
        st.session_state.processed_hashes = set()
    
    if 'processor' not in st.session_state:
        st.session_state.processor = None

def get_env_path(env_var: str, default_path: str) -> str:
    """Get path from environment variable or default"""
    return os.getenv(env_var, default_path)