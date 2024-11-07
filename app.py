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
        # Initialize the embeddings model with error handling
        # Try to load the model with error handling and fallback
        try:
            self.model = SentenceTransformer('sergeyzh/LaBSE-ru-turbo') 
        except Exception as e:
            st.warning(f"Failed to load primary model: {str(e)}. Trying fallback model...")
            try:
                # Try a smaller, more stable model as fallback
                self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            except Exception as e:
                st.error(f"Failed to load fallback model: {str(e)}")
                st.error("Please check your internet connection and try again.")
                st.stop()

        try:
            self.qdrant = QdrantClient(
                url=st.secrets["QDRANT_URL"],
                api_key=st.secrets["QDRANT_API_KEY"],
                timeout=60
            )
            
            # Check if collection exists
            collections = self.qdrant.get_collections()
            collection_exists = any(col.name == collection_name for col in collections.collections)
            
            if not collection_exists:
                # Create collection with cloud-appropriate settings
                self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=self.model.get_sentence_embedding_dimension(),
                        distance=models.Distance.COSINE
                    )
                )
                
                # Initialize metadata
                self._initialize_metadata()
            
            self.collection_name = collection_name
            self.processed_hashes = self._load_existing_hashes()
            
        except Exception as e:
            st.error(f"Error connecting to Qdrant: {str(e)}")
            st.stop()

    def _load_existing_hashes(self) -> Set[str]:
        """Load existing hashes from the database"""
        try:
            hashes = set()
            offset = None
            batch_size = 100

            while True:
                # Get batch of points
                response = self.qdrant.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True
                )
                
                # Check if we got any points
                if not response or not response[0]:
                    break
                    
                points, offset = response
                
                # Extract hashes from points
                for point in points:
                    if point.payload and 'hash' in point.payload:
                        hashes.add(point.payload['hash'])
                
                # If no offset returned, we've reached the end
                if not offset:
                    break

            return hashes
            
        except Exception as e:
            st.warning(f"Could not load existing hashes: {str(e)}")
            return set()
    def get_collection_stats(self) -> Dict:
        """Get current collection statistics"""
        try:
            # Count total points by scrolling through the collection
            total_points = 0
            offset = None
            batch_size = 100

            while True:
                response = self.qdrant.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True
                )
                
                if not response or not response[0]:
                    break
                    
                points, offset = response
                
                # Filter out metadata points in Python
                news_points = [p for p in points if not p.payload.get('is_metadata', False)]
                total_points += len(news_points)
                
                # Get metadata from the points if present
                metadata_points = [p for p in points if p.payload.get('is_metadata', False)]
                if metadata_points:
                    metadata = metadata_points[0].payload
                
                if not offset:
                    break

            # If we didn't find metadata in the points, use default values
            if not metadata_points:
                metadata = {'processed_files': [], 'last_updated': 'Never'}

            return {
                'total_points': total_points,
                'processed_files': len(metadata.get('processed_files', [])),
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
            offset = None
            metadata_id = 0
            current_metadata = None
            
            # Scroll through points to find metadata
            while True:
                response = self.qdrant.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True
                )
                
                if not response or not response[0]:
                    break
                    
                points, offset = response
                
                # Look for metadata point
                for point in points:
                    if point.payload.get('is_metadata', False):
                        current_metadata = point.payload
                        metadata_id = point.id
                        break
                
                if current_metadata or not offset:
                    break

            # Get existing processed files
            if current_metadata:
                processed_files = set(current_metadata.get('processed_files', []))
            else:
                processed_files = set()

            # Update metadata
            processed_files.add(file_name)
            
            new_metadata = {
                'is_metadata': True,
                'total_news': len(self.processed_hashes),
                'processed_files': list(processed_files),
                'last_updated': datetime.now().isoformat()
            }
            
            # Update metadata point
            self.qdrant.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=[
                    models.PointStruct(
                        id=metadata_id,
                        vector=[0.0] * self.model.get_sentence_embedding_dimension(),
                        payload=new_metadata
                    )
                ]
            )
            
        except Exception as e:
            st.warning(f"Could not update metadata: {str(e)}")

    def _initialize_metadata(self):
        """Initialize collection metadata"""
        try:
            self.qdrant.upsert(
                collection_name=self.collection_name,
                wait=True,  # Ensure metadata is written before continuing
                points=[
                    models.PointStruct(
                        id=0,
                        vector=[0.0] * self.model.get_sentence_embedding_dimension(),
                        payload={
                            'is_metadata': True,
                            'total_news': 0,
                            'processed_files': [],
                            'last_updated': datetime.now().isoformat()
                        }
                    )
                ]
            )
        except Exception as e:
            st.warning(f"Could not initialize metadata: {str(e)}")

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


    def _hash_news(self, company: str, date: datetime, text: str) -> str:
        """Create hash from news content to identify duplicates"""
        try:
            # Convert all inputs to strings and normalize them
            company_str = str(company).strip().lower()
            date_str = date.isoformat() if isinstance(date, datetime) else str(date)
            text_str = str(text).strip().lower()
            
            # Combine the fields to create a unique identifier
            content = f"{company_str}{date_str}{text_str}".encode('utf-8')
            
            # Create hash
            return hashlib.md5(content).hexdigest()
            
        except Exception as e:
            st.error(f"Error creating hash: {str(e)}")
            # Return a random hash to prevent processing failure
            return hashlib.md5(str(datetime.now().timestamp()).encode('utf-8')).hexdigest()
    

    def process_excel_file(self, file, similarity_threshold: float) -> int:
        """Process a single Excel file"""
        if self.is_file_processed(file.name):
            st.warning(f"File {file.name} was already processed. Skipping...")
            return 0
            
        try:
            # First check if the sheet exists
            try:
                xls = pd.ExcelFile(file)
                if 'Публикации' not in xls.sheet_names:
                    st.error(f"Sheet 'Публикации' not found in {file.name}")
                    return 0
            except Exception as e:
                st.error(f"Error reading {file.name}: {str(e)}")
                return 0

            # Read all columns first to check the structure
            try:
                df_all = pd.read_excel(
                    file,
                    sheet_name='Публикации',
                    header=None  # Read without header to get all rows
                )
                
                # Check if we have enough columns
                if df_all.shape[1] < 7:
                    st.error(f"File {file.name} doesn't have enough columns. Found {df_all.shape[1]}, need at least 7")
                    return 0
                
                # Now read only the columns we need
                df = pd.DataFrame({
                    'company': df_all.iloc[:, 0],  # Column A (index 0)
                    'date': df_all.iloc[:, 3],     # Column D (index 3)
                    'text': df_all.iloc[:, 6]      # Column G (index 6)
                })
                
                # Remove rows where all values are NaN (empty rows)
                df = df.dropna(how='all')
                
                # Show initial row count
                initial_count = len(df)
                st.info(f"Found {initial_count} rows in {file.name}")
                
                if df.empty:
                    st.warning(f"No data found in {file.name}")
                    return 0
                
                # Show data preview
                st.write("Preview of the first few rows:")
                st.write(df.head())
                
                # Clean data: remove rows where any required column is empty
                df = df.dropna(subset=['company', 'date', 'text'])
                after_cleaning_count = len(df)
                if after_cleaning_count < initial_count:
                    st.info(f"Removed {initial_count - after_cleaning_count} rows with missing data")
                
                # Convert dates with error handling
                try:
                    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M', errors='coerce')
                    invalid_dates = df['date'].isna().sum()
                    df = df.dropna(subset=['date'])
                    if invalid_dates > 0:
                        st.warning(f"Found {invalid_dates} rows with invalid dates, they will be skipped")
                except Exception as e:
                    st.error(f"Error converting dates: {str(e)}")
                    st.write("Date column preview:", df['date'].head())
                    return 0
                
                if df.empty:
                    st.warning("No valid data remains after cleaning")
                    return 0
                
                # Rest of the processing remains the same...
                [rest of your existing processing code]

            except Exception as e:
                st.error(f"Error reading Excel data: {str(e)}")
                return 0
                
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            return 0

    def search_news(self, query: str, company: str = None, limit: int = 10) -> List[Dict]:
        """Search news by text query and optionally filter by company"""
        try:
            query_vector = self.model.encode(query)
            
            # Search all points
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit * 2  # Get more results initially to allow for filtering
            )
            
            # Filter results in Python
            filtered_results = []
            for hit in results:
                # Skip metadata points
                if hit.payload.get('is_metadata', False):
                    continue
                    
                # Apply company filter if specified
                if company and company != "All Companies" and hit.payload['company'] != company:
                    continue
                    
                filtered_results.append({
                    'company': hit.payload['company'],
                    'date': hit.payload['date'],
                    'text': hit.payload['text'],
                    'similarity': hit.score,
                    'source_file': hit.payload.get('source_file', 'Unknown')
                })
                
                if len(filtered_results) >= limit:
                    break
            
            return filtered_results
            
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []

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