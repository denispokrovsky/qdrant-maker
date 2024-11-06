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

# Try importing required packages with error handling
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    st.error(f"""Error loading required packages. Please make sure all dependencies are installed:
    ```
    pip install -r requirements.txt
    ```
    Error details: {str(e)}
    """)
    st.stop()

    
# Initialize session state
if 'processed_hashes' not in st.session_state:
    st.session_state.processed_hashes = set()

if 'processor' not in st.session_state:
    st.session_state.processor = None

def get_env_path(env_var: str, default_path: str) -> str:
    """Get path from environment variable or default"""
    return os.getenv(env_var, default_path)

class NewsProcessor:
    def __init__(self, storage_path: str):
        """Initialize NewsProcessor with storage path"""
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize the embeddings model
        self.model = SentenceTransformer('sergeyzh/LaBSE-ru-turbo')
        
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
        """
        Process a single Excel file and store news in Qdrant
        Args:
            file: File object (can be path or file-like object)
        Returns:
            int: Number of processed news items
        """
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
                
                embedding = self.model.encode(row['text'])
                
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
        """
        Get list of unique companies in the database
        Returns:
            List[str]: Sorted list of company names
        """
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
        """
        Search news by text query and optionally filter by company
        Args:
            query (str): Search query
            company (str, optional): Company name to filter by
            limit (int): Maximum number of results
        Returns:
            List[Dict]: List of search results with company, date, text, and similarity score
        """
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

    def get_statistics(self) -> Dict:
        """
        Get database statistics
        Returns:
            Dict: Statistics about the database
        """
        try:
            scroll_result = self.qdrant.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                limit=None  # Get all records
            )
            
            records = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result
            
            stats = {
                'total_documents': len(records),
                'unique_companies': len(self.get_unique_companies()),
                'sources': set(),
                'date_range': {'earliest': None, 'latest': None}
            }
            
            for record in records:
                if 'source_file' in record.payload:
                    stats['sources'].add(record.payload['source_file'])
                
                date = datetime.fromisoformat(record.payload['date'])
                if stats['date_range']['earliest'] is None or date < stats['date_range']['earliest']:
                    stats['date_range']['earliest'] = date
                if stats['date_range']['latest'] is None or date > stats['date_range']['latest']:
                    stats['date_range']['latest'] = date
            
            stats['sources'] = sorted(stats['sources'])
            
            return stats
        except Exception as e:
            st.error(f"Error getting statistics: {str(e)}")
            return {}

    def clear_database(self):
        """Delete all data from the database"""
        try:
            self.qdrant.delete_collection(self.collection_name)
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.model.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE
                )
            )
            st.session_state.processed_hashes = set()
            return True
        except Exception as e:
            st.error(f"Error clearing database: {str(e)}")
            return False
    # Add all other methods from the NewsProcessor class here...
    # (process_excel_file, get_unique_companies, search_news, etc.)

# Get paths from environment variables or use defaults
NEWS_DIR = get_env_path("NEWS_DIR", "./news_files")
QDRANT_DIR = get_env_path("QDRANT_DIR", "./qdrant_storage")

def display_statistics(processor):
    """Display database statistics in the sidebar"""
    stats = processor.get_statistics()
    
    if stats:
        st.sidebar.header("Database Statistics")
        st.sidebar.metric("Total Documents", stats['total_documents'])
        st.sidebar.metric("Unique Companies", stats['unique_companies'])
        
        if stats['date_range']['earliest'] and stats['date_range']['latest']:
            st.sidebar.markdown("**Date Range:**")
            st.sidebar.text(f"From: {stats['date_range']['earliest'].strftime('%Y-%m-%d')}")
            st.sidebar.text(f"To: {stats['date_range']['latest'].strftime('%Y-%m-%d')}")
        
        if stats['sources']:
            with st.sidebar.expander("Source Files"):
                for source in stats['sources']:
                    st.text(source)

def main():
    st.set_page_config(page_title="News Clips Processor", layout="wide")
    
    st.title("News Clips Processor")
    
    # Ensure directories exist
    os.makedirs(QDRANT_DIR, exist_ok=True)
    os.makedirs(NEWS_DIR, exist_ok=True)
    
    # Initialize processor if needed
    if st.session_state.processor is None:
        try:
            st.session_state.processor = NewsProcessor(QDRANT_DIR)
            st.sidebar.success("Database connected successfully!")
        except Exception as e:
            st.error(f"Error initializing database: {str(e)}")
            st.stop()

    # Display database statistics
    display_statistics(st.session_state.processor)
    
    # Database management
    with st.sidebar.expander("Database Management"):
        if st.button("Clear Database", key="clear_db"):
            if st.session_state.processor.clear_database():
                st.success("Database cleared successfully!")
                st.experimental_rerun()
    
    # Main interface tabs
    tab1, tab2 = st.tabs(["Process Files", "Search News"])
    
    # File Processing Tab
    with tab1:
        st.header("Process Files")
        uploaded_files = upload_files()
        
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
            st.experimental_rerun()
    
    # Search Tab
    with tab2:
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