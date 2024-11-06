import streamlit as st
import os
from pathlib import Path
from datetime import datetime
import hashlib
from typing import List, Dict
import pandas as pd

# Display initial loading message
st.set_page_config(page_title="News Clips Processor", layout="wide")
st.title("News Clips Processor")

# Initialize session state
if 'processed_hashes' not in st.session_state:
    st.session_state.processed_hashes = set()

if 'processor' not in st.session_state:
    st.session_state.processor = None

# Try importing required packages with error handling
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    st.error(f"""Error loading required packages. Please check the error details and make sure all dependencies are installed:
    Error: {str(e)}
    
    Try running:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    """)
    st.stop()

def get_env_path(env_var: str, default_path: str) -> str:
    """Get path from environment variable or default"""
    return os.getenv(env_var, default_path)

class NewsProcessor:
    def __init__(self, storage_path: str):
        """Initialize NewsProcessor with storage path"""
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize the embeddings model with error handling
        try:
            st.info("Loading language model... This might take a few moments.")
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            st.success("Language model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading language model: {str(e)}")
            raise
        
        # Initialize Qdrant client
        qdrant_storage = Path(storage_path) / "qdrant_storage"
        self.qdrant = QdrantClient(path=str(qdrant_storage))
        
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
            st.info(f"Created new vector database at {qdrant_storage}")

    def _hash_news(self, company: str, date: datetime, text: str) -> str:
        """Create hash from news content to identify duplicates"""
        content = f"{company}{date.isoformat()}{text}".encode('utf-8')
        return hashlib.md5(content).hexdigest()
    
    def process_excel_file(self, file) -> int:
        """Process a single Excel file and return the number of processed news items"""
        try:
            df = pd.read_excel(
                file,
                sheet_name='Публикации',
                usecols="A,D,G",
                header=0
            )
            
            df.columns = ['company', 'date', 'text']
            df = df.dropna(subset=['company', 'date', 'text'])
            df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M', errors='coerce')
            
            processed_count = 0
            points = []
            
            for _, row in df.iterrows():
                news_hash = self._hash_news(row['company'], row['date'], row['text'])
                
                if news_hash in st.session_state.processed_hashes:
                    continue
                
                # Generate embedding with error handling
                try:
                    embedding = self.model.encode(row['text'])
                except Exception as e:
                    st.warning(f"Error generating embedding for text: {str(e)}")
                    continue
                
                points.append(models.PointStruct(
                    id=len(st.session_state.processed_hashes),
                    vector=embedding.tolist(),
                    payload={
                        'company': row['company'],
                        'date': row['date'].isoformat(),
                        'text': row['text'],
                        'source_file': getattr(file, 'name', str(file)),
                        'hash': news_hash,
                        'processed_date': datetime.now().isoformat()
                    }
                ))
                
                st.session_state.processed_hashes.add(news_hash)
                processed_count += 1
                
                if len(points) >= 100:
                    self.qdrant.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    points = []
            
            if points:
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
            
            return processed_count
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return 0

    def get_unique_companies(self) -> List[str]:
        """Get list of unique companies in the database"""
        try:
            scroll_result = self.qdrant.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                limit=1000
            )
            
            records = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result
            company_names = set()
            
            for record in records:
                if "company" in record.payload:
                    company_names.add(record.payload["company"])
            
            return sorted(company_names)
        except Exception as e:
            st.error(f"Error fetching companies: {str(e)}")
            return []

    def search_news(self, query: str, company: str = None, limit: int = 10) -> List[Dict]:
        """Search news by text query and optionally filter by company"""
        try:
            query_vector = self.model.encode(query)
            
            filter_conditions = None
            if company and company != "All Companies":
                filter_conditions = models.Filter(
                    must=[models.FieldCondition(
                        key="company",
                        match=models.MatchValue(value=company)
                    )]
                )
            
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=filter_conditions
            )
            
            return [{
                'company': hit.payload['company'],
                'date': hit.payload['date'],
                'text': hit.payload['text'],
                'similarity': hit.score,
                'source_file': hit.payload.get('source_file', 'Unknown'),
                'processed_date': hit.payload.get('processed_date', 'Unknown')
            } for hit in results]
        except Exception as e:
            st.error(f"Error performing search: {str(e)}")
            return []

# Get paths from environment variables or use defaults
NEWS_DIR = get_env_path("NEWS_DIR", "./news_files")
QDRANT_DIR = get_env_path("QDRANT_DIR", "./qdrant_storage")

def main():
    st.sidebar.header("Configuration")
    st.sidebar.info(f"Database location: {QDRANT_DIR}")
    
    # Initialize processor
    if st.session_state.processor is None:
        try:
            st.session_state.processor = NewsProcessor(QDRANT_DIR)
        except Exception as e:
            st.error(f"Error initializing processor: {str(e)}")
            st.stop()
    
    # File upload section
    st.header("Upload and Process Files")
    uploaded_files = st.file_uploader(
        "Choose Excel files", 
        type="xlsx",
        accept_multiple_files=True
    )
    
    if uploaded_files:
        total_processed = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Processing {file.name}...")
            processed_count = st.session_state.processor.process_excel_file(file)
            total_processed += processed_count
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("Processing complete!")
        st.success(f"Processed {len(uploaded_files)} files. Added {total_processed} unique news items.")
    
    # Search section
    st.header("Search News")
    companies = ["All Companies"] + st.session_state.processor.get_unique_companies()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input("Search Query", "")
    
    with col2:
        selected_company = st.selectbox("Filter by Company", companies)
    
    num_results = st.slider("Number of results", min_value=1, max_value=20, value=5)
    
    if st.button("Search"):
        if not search_query:
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching..."):
                results = st.session_state.processor.search_news(
                    query=search_query,
                    company=selected_company,
                    limit=num_results
                )
            
            if not results:
                st.info("No results found.")
            else:
                for i, result in enumerate(results, 1):
                    with st.expander(f"Result {i} - {result['company']} ({result['date'][:10]})"):
                        st.markdown(f"**Relevance Score:** {result['similarity']:.2f}")
                        st.markdown(f"**Company:** {result['company']}")
                        st.markdown(f"**Date:** {result['date']}")
                        st.markdown(f"**Source:** {result['source_file']}")
                        st.markdown(f"**Processed:** {result['processed_date']}")
                        st.markdown("**Text:**")
                        st.text(result['text'])

if __name__ == "__main__":
    main()