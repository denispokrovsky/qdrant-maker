import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import hashlib
from typing import List, Dict, Set
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from tqdm import tqdm

# Page config
st.set_page_config(
    page_title="News Processor",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    .stAlert > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stExpander {
        background-color: #F0F2F6;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = 0
if 'total_news' not in st.session_state:
    st.session_state.total_news = 0

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    similarity_threshold = st.slider(
        "Deduplication Similarity Threshold (%)",
        min_value=50,
        max_value=100,
        value=65,
        help="Higher values mean more strict deduplication"
    )
    
    st.markdown("---")
    st.markdown("""
        ### Instructions
        1. Upload Excel files in the Process tab
        2. Wait for processing to complete
        3. Switch to Search tab to find news
        
        ### File Requirements
        Excel files should have:
        - Sheet named 'Публикации'
        - Company names in Column A
        - Dates in Column D (DD.MM.YYYY HH:MM)
        - News text in Column G
    """)

class NewsProcessor:
    def __init__(self, collection_name: str = "company_news"):
        self.model = SentenceTransformer('sergeyzh/LaBSE-ru-turbo')
        
        # Initialize Qdrant client with cloud credentials
        try:
            self.qdrant = QdrantClient(
                url=st.secrets["QDRANT_URL"],
                api_key=st.secrets["QDRANT_API_KEY"]
            )
            
            # Check if collection exists; create if not
            if not self.qdrant.collection_exists(collection_name):
                self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=self.model.get_sentence_embedding_dimension(),
                        distance=models.Distance.COSINE
                    )
                )
            self.collection_name = collection_name
            self.processed_hashes = set()
            
        except Exception as e:
            st.error(f"Failed to connect to Qdrant Cloud: {str(e)}")
            st.stop()

    def _hash_news(self, company: str, date: datetime, text: str) -> str:
        content = f"{company}{date.isoformat()}{text}".encode('utf-8')
        return hashlib.md5(content).hexdigest()

    def fuzzy_deduplicate(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        with st.spinner("Deduplicating entries..."):
            seen_texts = []
            indices_to_keep = []
            
            progress_bar = st.progress(0)
            total_rows = len(df)
            
            for i, row in enumerate(df.iterrows()):
                text = str(row[1]['text'])
                if pd.isna(text):
                    indices_to_keep.append(row[0])
                    continue
                    
                if not seen_texts or all(fuzz.ratio(text, seen) < threshold for seen in seen_texts):
                    seen_texts.append(text)
                    indices_to_keep.append(row[0])
                
                progress_bar.progress((i + 1) / total_rows)
            
            return df.iloc[indices_to_keep]

    def process_excel_file(self, file, similarity_threshold: float) -> int:
        try:
            # Read Excel file
            df = pd.read_excel(
                file,
                sheet_name='Публикации',
                usecols="A,D,G",
                header=0
            )
            
            # Rename columns
            df.columns = ['company', 'date', 'text']
            
            # Clean data
            df = df.dropna(subset=['company', 'date', 'text'])
            df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M', errors='coerce')
            
            # Deduplicate
            original_count = len(df)
            df = self.fuzzy_deduplicate(df, similarity_threshold)
            deduped_count = len(df)
            
            st.info(f"Removed {original_count - deduped_count} duplicate entries from {file.name}")
            
            # Process entries
            processed_count = 0
            points = []
            
            progress_bar = st.progress(0)
            
            for i, row in enumerate(df.iterrows()):
                row = row[1]  # Get the row data
                news_hash = self._hash_news(row['company'], row['date'], row['text'])
                
                if news_hash in self.processed_hashes:
                    continue
                    
                embedding = self.model.encode(row['text'])
                
                points.append(models.PointStruct(
                    id=len(self.processed_hashes),
                    vector=embedding.tolist(),
                    payload={
                        'company': row['company'],
                        'date': row['date'].isoformat(),
                        'text': row['text'],
                        'source_file': file.name,
                        'hash': news_hash
                    }
                ))
                
                self.processed_hashes.add(news_hash)
                processed_count += 1
                
                if len(points) >= 100:
                    self.qdrant.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    points = []
                
                progress_bar.progress((i + 1) / len(df))
            
            if points:
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
            
            return processed_count
            
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            return 0

    def search_news(self, query: str, company: str = None, limit: int = 10) -> List[Dict]:
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
            'similarity': hit.score
        } for hit in results]

def main():
    st.title("📰 News Processor and Search")
    
    tab1, tab2 = st.tabs(["Process Files", "Search News"])
    
    with tab1:
        st.header("Process Excel Files")
        uploaded_files = st.file_uploader(
            "Choose Excel files",
            type=['xlsx'],
            accept_multiple_files=True,
            help="Select one or more Excel files to process"
        )
        
        if uploaded_files:
            processor = NewsProcessor()
            
            with st.spinner("Processing files..."):
                for file in uploaded_files:
                    st.subheader(f"Processing {file.name}")
                    
                    processed_count = processor.process_excel_file(
                        file,
                        similarity_threshold
                    )
                    
                    st.session_state.processed_files += 1
                    st.session_state.total_news += processed_count
            
            st.success(f"""
                Processing complete! 🎉
                - Files processed: {st.session_state.processed_files}
                - Total unique news items: {st.session_state.total_news}
            """)
    
    with tab2:
        st.header("Search News")
        
        try:
            processor = NewsProcessor()
            
            # Get unique companies
            scroll_result = processor.qdrant.scroll(
                collection_name=processor.collection_name,
                with_payload=True,
                limit=1000
            )
            records = scroll_result[0]
            companies = sorted(set(record.payload["company"] for record in records))
            
            # Search interface
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                search_query = st.text_input(
                    "Search Query",
                    "",
                    help="Enter keywords to search for in news articles"
                )
            
            with col2:
                selected_company = st.selectbox(
                    "Filter by Company",
                    ["All Companies"] + companies
                )
            
            with col3:
                num_results = st.number_input(
                    "Max Results",
                    min_value=1,
                    max_value=50,
                    value=5
                )
            
            if st.button("🔍 Search", type="primary") and search_query:
                with st.spinner("Searching..."):
                    results = processor.search_news(
                        search_query,
                        selected_company,
                        num_results
                    )
                    
                    if results:
                        for i, result in enumerate(results, 1):
                            with st.expander(
                                f"📄 Result {i}: {result['company']} ({result['date'][:10]}) - Relevance: {result['similarity']:.2f}"
                            ):
                                st.markdown(f"**Company:** {result['company']}")
                                st.markdown(f"**Date:** {result['date']}")
                                st.markdown("**Text:**")
                                st.text(result['text'])
                    else:
                        st.info("No results found.")
                        
        except Exception as e:
            st.error(f"Error connecting to the database: {str(e)}")
            st.info("Please make sure the database is properly configured.")

if __name__ == "__main__":
    main()