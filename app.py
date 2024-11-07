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

class NewsProcessor:
    def __init__(self, collection_name: str = "company_news"):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        try:
            self.qdrant = QdrantClient(
                url=st.secrets["QDRANT_URL"],
                api_key=st.secrets["QDRANT_API_KEY"]
            )
            
            # Create collection if it doesn't exist
            if not self.qdrant.collection_exists(collection_name):
                self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=self.model.get_sentence_embedding_dimension(),
                        distance=models.Distance.COSINE
                    )
                )
                
                # Initialize collection metadata
                self.qdrant.set_payload(
                    collection_name=collection_name,
                    payload={
                        'total_news': 0,
                        'processed_files': set(),
                        'last_updated': datetime.now().isoformat()
                    },
                    points=[0]  # Metadata point
                )
            
            self.collection_name = collection_name
            
            # Load existing hashes from the database
            self.processed_hashes = self._load_existing_hashes()
            
        except Exception as e:
            st.error(f"Failed to connect to Qdrant Cloud: {str(e)}")
            st.stop()

    def _load_existing_hashes(self) -> Set[str]:
        """Load existing hashes from the database"""
        try:
            # Scroll through all points to get hashes
            hashes = set()
            offset = None
            limit = 100
            
            while True:
                scroll_response = self.qdrant.scroll(
                    collection_name=self.collection_name,
                    with_payload=True,
                    limit=limit,
                    offset=offset
                )
                
                records, offset = scroll_response
                
                if not records:
                    break
                    
                for record in records:
                    if 'hash' in record.payload:
                        hashes.add(record.payload['hash'])
            
            return hashes
            
        except Exception as e:
            st.warning(f"Could not load existing hashes: {str(e)}")
            return set()

    def get_collection_stats(self) -> Dict:
        """Get current collection statistics"""
        try:
            collection_info = self.qdrant.get_collection(self.collection_name)
            points_count = collection_info.points_count
            
            # Get metadata
            metadata = self.qdrant.retrieve(
                collection_name=self.collection_name,
                ids=[0],
                with_payload=True
            )[0].payload
            
            return {
                'total_points': points_count,
                'processed_files': len(metadata.get('processed_files', set())),
                'last_updated': metadata.get('last_updated', 'Never')
            }
        except Exception as e:
            st.warning(f"Could not retrieve collection stats: {str(e)}")
            return {
                'total_points': 0,
                'processed_files': 0,
                'last_updated': 'Never'
            }

    def _update_metadata(self, file_name: str):
        """Update collection metadata after processing a file"""
        try:
            # Get current metadata
            metadata = self.qdrant.retrieve(
                collection_name=self.collection_name,
                ids=[0],
                with_payload=True
            )[0].payload
            
            # Update metadata
            processed_files = set(metadata.get('processed_files', set()))
            processed_files.add(file_name)
            
            new_metadata = {
                'total_news': len(self.processed_hashes),
                'processed_files': list(processed_files),  # Convert to list for JSON serialization
                'last_updated': datetime.now().isoformat()
            }
            
            # Update metadata point
            self.qdrant.set_payload(
                collection_name=self.collection_name,
                payload=new_metadata,
                points=[0]
            )
            
        except Exception as e:
            st.warning(f"Could not update metadata: {str(e)}")

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

    def is_file_processed(self, file_name: str) -> bool:
        """Check if a file has already been processed"""
        try:
            metadata = self.qdrant.retrieve(
                collection_name=self.collection_name,
                ids=[0],
                with_payload=True
            )[0].payload
            
            processed_files = set(metadata.get('processed_files', set()))
            return file_name in processed_files
            
        except Exception:
            return False

    def process_excel_file(self, file, similarity_threshold: float) -> int:
        # Check if file was already processed
        if self.is_file_processed(file.name):
            st.warning(f"File {file.name} was already processed. Skipping...")
            return 0
            
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
                row = row[1]
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
            
            # Update metadata after successful processing
            self._update_metadata(file.name)
            
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
            'similarity': hit.score,
            'source_file': hit.payload.get('source_file', 'Unknown')
        } for hit in results]

def main():
    st.title("📰 News Processor and Search")
    
    # Initialize processor
    processor = NewsProcessor()
    
    # Get current stats
    stats = processor.get_collection_stats()
    
    # Display stats in sidebar
    with st.sidebar:
        st.title("⚙️ Settings & Stats")
        
        st.markdown("### Database Stats")
        st.write(f"Total News Items: {stats['total_points']}")
        st.write(f"Processed Files: {stats['processed_files']}")
        st.write(f"Last Updated: {stats['last_updated']}")
        
        st.markdown("---")
        
        similarity_threshold = st.slider(
            "Deduplication Similarity Threshold (%)",
            min_value=50,
            max_value=100,
            value=65,
            help="Higher values mean more strict deduplication"
        )
    
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
            with st.spinner("Processing files..."):
                for file in uploaded_files:
                    st.subheader(f"Processing {file.name}")
                    
                    processed_count = processor.process_excel_file(
                        file,
                        similarity_threshold
                    )
                    
                    if processed_count > 0:
                        st.success(f"Added {processed_count} new news items from {file.name}")
            
            # Refresh stats after processing
            stats = processor.get_collection_stats()
            st.success(f"""
                Processing complete! 🎉
                - Total news items in database: {stats['total_points']}
                - Total files processed: {stats['processed_files']}
                - Last updated: {stats['last_updated']}
            """)
    
    with tab2:
        st.header("Search News")
        
        try:
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
                                st.markdown(f"**Source File:** {result['source_file']}")
                                st.markdown("**Text:**")
                                st.text(result['text'])
                    else:
                        st.info("No results found.")
                        
        except Exception as e:
            st.error(f"Error connecting to the database: {str(e)}")
            st.info("Please make sure the database is properly configured.")

if __name__ == "__main__":
    main()