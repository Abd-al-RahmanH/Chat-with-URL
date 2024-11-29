from urllib.parse import urlparse
from dotenv import load_dotenv
import os
import chromadb
import streamlit as st
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_credentials():
    load_dotenv()
    api_key = os.getenv("api_key", None)
    watsonx_project_id = os.getenv("project_id", None)

    if not api_key or not watsonx_project_id:
        raise EnvironmentError("Missing `api_key` or `project_id` in the .env file.")
    
    return api_key, watsonx_project_id

def load_css(file_name):
    try:
        with open(file_name) as file:
            st.markdown(f'<style>{file.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file '{file_name}' not found.")
    except Exception as e:
        st.error(f"Failed to load CSS file: {e}")

def create_collection_name(url):
    try:
        parsed_url = urlparse(url)
        domain_parts = parsed_url.netloc.split('.')
        if len(domain_parts) >= 2:
            return domain_parts[-2]
        else:
            return "default_collection"
    except Exception as e:
        logger.warning(f"Invalid URL '{url}': {e}")
        return "invalid_url"

def chromadb_client():
    from chromadb.config import Settings
    
    current_dir = os.getcwd()
    custom_cache_path = os.path.join(current_dir, ".cache")
    
    try:
        settings = Settings(persist_directory=custom_cache_path)
        client = chromadb.Client(settings)
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ChromaDB client: {e}")

def clear_collection(collection_name, client):
    try:
        collection = client.get_collection(collection_name)
        if collection:
            collection.delete()
            logger.info(f"Collection '{collection_name}' cleared successfully!")
    except ValueError:
        logger.warning(f"Collection '{collection_name}' does not exist, skipping.")
    except Exception as e:
        logger.error(f"Failed to clear collection '{collection_name}': {e}")
