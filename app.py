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
    page_title="обработка новостного потока для базы поиска фактов",
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
        """Get collection statistics including size"""
        try:
            total_points = 0
            total_size_bytes = 0
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
                
                # Filter out metadata points
                news_points = [p for p in points if not p.payload.get('is_metadata', False)]
                total_points += len(news_points)
                
                # Calculate size for each point
                for point in news_points:
                    # Vector size (float32 * vector dimension)
                    vector_size = len(point.vector) * 4
                    # Payload size (rough estimation)
                    payload_size = len(str(point.payload).encode('utf-8'))
                    total_size_bytes += vector_size + payload_size
                
                if not offset:
                    break

            # Convert bytes to GB
            size_gb = total_size_bytes / (1024 * 1024 * 1024)

            return {
                'total_points': total_points,
                'size_gb': round(size_gb, 2),
                'processed_files': len(self.processed_hashes),
                'last_updated': datetime.now().isoformat()
            }
                    
        except Exception as e:
            st.warning(f"Could not retrieve collection stats: {str(e)}")
            return {
                'total_points': 0,
                'size_gb': 0,
                'processed_files': 0,
                'last_updated': 'Never'
            }

    # Then in main(), update the sidebar display:
    with st.sidebar:
        st.title("⚙️ Settings & Stats")
        
        st.markdown("### Database Stats")
        st.write(f"всего новостей: {stats['total_points']}")
        st.write(f"файлов обработано: {stats['processed_files']}")
        st.write(f"Размер базы: {stats['size_gb']} GB")
        st.write(f"Обновлено: {stats['last_updated']}")
        
        st.markdown("---")


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
        """Deduplicate rows using fuzzy matching with safer index handling"""
        with st.spinner("Deduplicating entries..."):
            # Reset index to ensure continuous integers
            df = df.reset_index(drop=True)
            
            seen_texts = []
            indices_to_keep = []
            
            progress_bar = st.progress(0)
            total_rows = len(df)
            
            # Debug information
            st.write(f"Starting deduplication of {total_rows} rows")
            
            try:
                for idx, row in df.iterrows():
                    try:
                        # Safely get text value
                        text = str(row.get('text', ''))
                        
                        # Handle empty or NaN text
                        if pd.isna(text) or not text.strip():
                            indices_to_keep.append(idx)
                            continue
                        
                        # Check for fuzzy duplicates
                        if not seen_texts or all(fuzz.ratio(text, seen) < threshold for seen in seen_texts):
                            seen_texts.append(text)
                            indices_to_keep.append(idx)
                        
                        # Update progress
                        if idx % 10 == 0:  # Update every 10 rows to reduce overhead
                            progress_bar.progress((idx + 1) / total_rows)
                            
                    except Exception as e:
                        st.warning(f"Error processing row {idx}: {str(e)}")
                        continue
                
                # Final progress update
                progress_bar.progress(1.0)
                
                # Validate indices before using them
                valid_indices = [idx for idx in indices_to_keep if idx < len(df)]
                
                # Debug information
                st.write(f"Found {len(valid_indices)} unique rows after deduplication")
                
                # Return deduplicated dataframe
                return df.loc[valid_indices].reset_index(drop=True)
                
            except Exception as e:
                st.error(f"Error during deduplication: {str(e)}")
                # Return original dataframe if deduplication fails
                return df

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

            # Read Excel file - simpler approach
            try:
                df = pd.read_excel(
                    file,
                    sheet_name='Публикации',
                    usecols="A, D, G",  # Using column indices instead of letters
                    names=['company', 'date', 'text'],  # Assign column names directly
                    dtype=str  # Read everything as string initially
                )
                
                # Debug info
                st.write("File structure:")
                st.write(f"Columns: {df.columns.tolist()}")
                st.write(f"Total rows before cleaning: {len(df)}")
                
                # Remove rows where all values are NaN (empty rows)
                df = df.dropna(how='all')
                
                # Show initial row count
                initial_count = len(df)
                st.info(f"Found {initial_count} rows in {file.name}")


                # Show data preview
                st.write("Preview of raw data:")
                st.write(df.head())
                
                # Clean data: remove rows where any required column is empty
                df = df.dropna(subset=['company', 'text'])
                after_cleaning_count = len(df)
                if after_cleaning_count < initial_count:
                    st.info(f"Removed {initial_count - after_cleaning_count} rows with missing data")
                

                # no need in parsing!
                
                df['parsed_date'] = pd.to_datetime(df['date']).dt.strftime('%d.%m.%Y')
                
                st.write(df['parsed_date'].head(5))

                             
                # Count invalid dates
                invalid_dates = df['parsed_date'].isna()
                if invalid_dates.any():
                    st.info(f"Found {invalid_dates.sum()} rows with unparseable dates - these will be kept with empty date values")
                
                invalid_dates = df['date'].isna().sum()
                if invalid_dates > 0:
                    st.info(f"Found {invalid_dates} rows with unparseable dates - these will be kept with empty date values")

                # Deduplicate within file
                st.write("starting deduplication")
                original_count = len(df)
                st.write(f"original_count {original_count}")
                df = self.fuzzy_deduplicate(df, similarity_threshold)
                st.write(f"success dedup")
                deduped_count = len(df)
                
                st.info(f"Removed {original_count - deduped_count} duplicate entries from {file.name}")
                
                # Process entries
                processed_count = 0
                points = []
                
                progress_bar = st.progress(0)
                
                for i, row in enumerate(df.iterrows()):
                    row = row[1]  # Get the row data
                    
                    try:
                        date_value = row['parsed_date'] if pd.notna(row['parsed_date']) else ''
                        
                        news_hash = self._hash_news(row['company'], date_value, row['text'])
                        
                        if news_hash in self.processed_hashes:
                            continue
                            
                        embedding = self.model.encode(row['text'])
                        
                        points.append(models.PointStruct(
                            id=len(self.processed_hashes),
                            vector=embedding.tolist(),
                            payload={
                                'company': row['company'],
                                'date': row['parsed_date'],
                                'text': row['text'],
                                'source_file': file.name,
                                'hash': news_hash
                            }
                        ))
                        
                        self.processed_hashes.add(news_hash)
                        processed_count += 1
                        
                        # Batch insert
                        if len(points) >= 20:
                            self.qdrant.upsert(
                                collection_name=self.collection_name,
                                wait=True,
                                points=points
                            )
                            points = []
                            
                    except Exception as e:
                        st.warning(f"Error processing row {i + 1}: {str(e)}")
                        continue
                    
                    progress_bar.progress((i + 1) / len(df))
                
                # Insert remaining points
                if points:
                    self.qdrant.upsert(
                        collection_name=self.collection_name,
                        wait=True,
                        points=points
                    )
                
                # Update metadata if we processed any points
                if processed_count > 0:
                    self._update_metadata(file.name)
                
                return processed_count
                
            except Exception as e:
                st.error(f"Error reading Excel data: {str(e)}")
                # Add more debug information
                st.write("Trying to read file:", file.name)
                st.write("Available sheets:", xls.sheet_names)
                return 0
                
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            return 0

    def search_news(self, query: str, company: str = None, limit: int = 10) -> List[Dict]:
        """Enhanced search with fixed company filtering"""
        try:
            # Get more initial results to allow for filtering
            initial_limit = limit * 5
            query_vector = self.model.encode(query)
            
            # Perform vector search
            vector_results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=initial_limit,
                with_payload=True,
                search_params=models.SearchParams(
                    hnsw_ef=128,
                    exact=False
                )
            )
            
            processed_results = []
            
            for hit in vector_results:
                # Skip metadata points
                if hit.payload.get('is_metadata', False):
                    continue
                    
                # Fix company filtering
                if company and company != "Все компании" and hit.payload.get('company', '').strip() != company.strip():
                    continue
                
                text = hit.payload.get('text', '').lower()
                query_lower = query.lower()
                
                # Calculate scores
                vector_score = hit.score
                fuzzy_ratio = fuzz.ratio(query_lower, text) / 100
                partial_ratio = fuzz.partial_ratio(query_lower, text) / 100
                token_sort_ratio = fuzz.token_sort_ratio(query_lower, text) / 100
                contains_exact = query_lower in text
                
                # Adjust score weights for Russian text
                combined_score = (
                    vector_score * 0.5 +       # Increased vector importance
                    fuzzy_ratio * 0.15 +
                    partial_ratio * 0.25 +     # Increased partial match importance
                    token_sort_ratio * 0.1
                )
                
                # Boost exact matches more significantly
                if contains_exact:
                    combined_score *= 1.3
                
                # Lower threshold for better recall
                if combined_score > 0.2:  # Lowered threshold
                    processed_results.append({
                        'company': hit.payload.get('company', 'Неизвестно'),
                        'date': hit.payload.get('date', ''),
                        'text': hit.payload.get('text', ''),
                        'source_file': hit.payload.get('source_file', 'Неизвестно'),
                        'similarity': combined_score,
                        'vector_score': vector_score,
                        'fuzzy_score': fuzzy_ratio,
                        'partial_score': partial_ratio,
                        'token_score': token_sort_ratio,
                        'has_exact_match': contains_exact
                    })
            
            # Sort by combined score
            processed_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Take top results
            final_results = processed_results[:limit]
            
            return final_results
            
        except Exception as e:
            st.error(f"Ошибка поиска: {str(e)}")
            return []
    
def main():
    st.title("📰обработка новостного массива для исследования фактов")
    
    # Initialize processor
    processor = NewsProcessor()
    
    # Get current stats
    stats = processor.get_collection_stats()
    
    # Display stats in sidebar
    with st.sidebar:
        st.title("⚙️ Статистика & Настройки")
        
        st.markdown("### Статистика БД")
        st.write(f"всего новостей: {stats['total_points']}")
        st.write(f"файлов обработано: {stats['processed_files']}")
        st.write(f"Обновлено: {stats['last_updated']}")
        
        st.markdown("---")
        
        similarity_threshold = st.slider(
            "Deduplication Similarity Threshold (%)",
            min_value=50,
            max_value=100,
            value=65,
            help="Higher values mean more strict deduplication"
        )
    
    tab1, tab2 = st.tabs(["Обработать файлы", "Искать по новостям"])
    
    with tab1:
        st.header("Обработка и перекодирование Excel-файлов")
        uploaded_files = st.file_uploader(
            "Выбирай excel-файл",
            type=['xlsx'],
            accept_multiple_files=True,
            help="Select one or more Excel files to process"
        )
        
        if uploaded_files:
            with st.spinner("Обрабатываю файлы..."):
                for file in uploaded_files:
                    st.subheader(f"Обрабатываю файл {file.name}")
                    
                    processed_count = processor.process_excel_file(
                        file,
                        similarity_threshold
                    )
                    
                    if processed_count > 0:
                        st.success(f"Добавил {processed_count} сообщений {file.name}")
            
            # Refresh stats after processing
            stats = processor.get_collection_stats()
            st.success(f"""
                Готово! 🎉
                - Всего новостей в БД: {stats['total_points']}
                - Файлов обработано: {stats['processed_files']}
                - Обновлено: {stats['last_updated']}
            """)
    with tab2:
        st.header("Поиск по новостям")
        
        try:
            # Get unique companies
            companies = set()
            offset = None
            
            while True:
                scroll_result = processor.qdrant.scroll(
                    collection_name=processor.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True
                )
                
                if not scroll_result or not scroll_result[0]:
                    break
                    
                points, offset = scroll_result
                
                # Extract companies, clean them up
                for point in points:
                    if point.payload and not point.payload.get('is_metadata'):
                        company = point.payload.get('company', '').strip()
                        if company:
                            companies.add(company)
                
                if not offset:
                    break
            
            companies = sorted(companies)
            
            if not companies:
                st.warning("Компании не найдены в базе")
                return
            
            # Search interface
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                search_query = st.text_input(
                    "Поисковый запрос",
                    "",
                    help="Введите ключевые слова для поиска"
                )
            
            with col2:
                selected_company = st.selectbox(
                    "Компания",
                    ["Все компании"] + companies
                )
            
            with col3:
                num_results = st.number_input(
                    "Макс. результатов",
                    min_value=1,
                    max_value=50,
                    value=10
                )
            
            if st.button("🔍 Искать", key="search_button") and search_query:
                with st.spinner("Ищу..."):
                    results = processor.search_news(
                        search_query,
                        selected_company,
                        num_results
                    )
                    
                    if results:
                        st.success(f"Найдено {len(results)} результатов")
                        
                        for i, result in enumerate(results, 1):
                            with st.expander(
                                f"📄 {i}. {result['company']} - {result.get('date', 'дата не указана')} (релевантность: {result['similarity']:.2f})"
                            ):
                                cols = st.columns([2, 1])
                                with cols[0]:
                                    st.markdown("**Текст:**")
                                    st.text(result['text'])
                                
                                with cols[1]:
                                    st.markdown("**Детали:**")
                                    st.write(f"Компания: {result['company']}")
                                    st.write(f"Дата: {result['date']}")
                                    st.write(f"Файл: {result['source_file']}")
                                    
                                    if st.checkbox(f"Показать метрики {i}", key=f"debug_{i}"):
                                        st.write("Оценки:")
                                        st.write(f"Векторная: {result.get('vector_score', 0):.3f}")
                                        st.write(f"Нечеткая: {result.get('fuzzy_score', 0):.3f}")
                                        st.write(f"Частичная: {result.get('partial_score', 0):.3f}")
                                        st.write(f"Токены: {result.get('token_score', 0):.3f}")
                                        st.write(f"Точное совпадение: {'Да' if result.get('has_exact_match', False) else 'Нет'}")
                    else:
                        st.info("Ничего не найдено")
                        
        except Exception as e:
            st.error(f"Ошибка подключения к базе данных: {str(e)}")
            st.info("Проверьте настройки подключения к базе данных.")

if __name__ == "__main__":
    main()