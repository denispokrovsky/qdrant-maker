import streamlit as st
import os
from src.processor import NewsProcessor
from src.utils import initialize_session_state, get_env_path
from pathlib import Path
from datetime import datetime

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

def upload_files():
    """Handle file uploads"""
    uploaded_files = st.file_uploader(
        "Choose Excel files", 
        type="xlsx",
        accept_multiple_files=True
    )
    return uploaded_files

def main():
    st.set_page_config(page_title="News Clips Processor", layout="wide")
    
    st.title("News Clips Processor")
    
    # Initialize session state
    initialize_session_state()
    
    # Ensure directories exist
    os.makedirs(QDRANT_DIR, exist_ok=True)
    os.makedirs(NEWS_DIR, exist_ok=True)
    
    # Display configured paths
    st.sidebar.header("Configuration")
    st.sidebar.info(f"Database location: {QDRANT_DIR}")
    
    # Initialize processor
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