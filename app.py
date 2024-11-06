import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import hashlib
from typing import List, Dict
import os

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# Add storage path configuration
def initialize_storage_settings():
    if 'storage_path' not in st.session_state:
        # Default to user's documents folder
        default_path = os.path.join(os.path.expanduser("~"), "Documents", "NewsDatabase")
        st.session_state.storage_path = default_path

    if 'processed_hashes' not in st.session_state:
        st.session_state.processed_hashes = set()
    
    if 'processor' not in st.session_state:
        st.session_state.processor = None

class NewsProcessor:
    def __init__(self, storage_path: str):
        """
        Initialize NewsProcessor with custom storage path
        """
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize the embeddings model
        self.model = SentenceTransformer('sergeyzh/LaBSE-ru-turbo')
        
        # Initialize Qdrant client with custom path
        qdrant_path = os.path.join(self.storage_path, "qdrant_storage")
        self.qdrant = QdrantClient(path=qdrant_path)
        
        # Collection configuration
        self.collection_name = "company_news"
        if not self.qdrant.collection_exists(self.collection_name):
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.model.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE
                )
            )

    # ... rest of the NewsProcessor methods remain the same ...

def main():
    st.set_page_config(page_title="News Clips Processor", layout="wide")
    
    st.title("News Clips Processor")
    
    # Initialize storage settings
    initialize_storage_settings()
    
    # Storage configuration section
    with st.expander("Storage Settings"):
        new_path = st.text_input(
            "Database Storage Location",
            value=st.session_state.storage_path,
            help="Specify the full path where the database will be stored"
        )
        
        if new_path != st.session_state.storage_path:
            if st.button("Update Storage Location"):
                try:
                    # Create directory if it doesn't exist
                    os.makedirs(new_path, exist_ok=True)
                    st.session_state.storage_path = new_path
                    # Reset processor to use new location
                    st.session_state.processor = None
                    st.success(f"Storage location updated to: {new_path}")
                    st.info("Please note that this creates a new database. Previous data will remain in the old location.")
                except Exception as e:
                    st.error(f"Error setting storage location: {str(e)}")
    
    # Initialize processor if needed
    if st.session_state.processor is None:
        try:
            st.session_state.processor = NewsProcessor(st.session_state.storage_path)
        except Exception as e:
            st.error(f"Error initializing database: {str(e)}")
            st.stop()
    
    # Display current storage location
    st.info(f"Current database location: {st.session_state.storage_path}")
    
    # Rest of your app code (tabs, processing, search) remains the same...

if __name__ == "__main__":
    main()