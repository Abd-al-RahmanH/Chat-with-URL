from urllib.parse import urlparse
from dotenv import load_dotenv
import os
import chromadb
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Save logs to a file
        logging.StreamHandler()         # Show logs in the console
    ]
)
logger = logging.getLogger(__name__)

def get_credentials():
    """Load credentials from the .env file."""
    load_dotenv()
    api_key = os.getenv("api_key", None)
    watsonx_project_id = os.getenv("project_id", None)

    if not api_key or not watsonx_project_id:
        raise EnvironmentError("Missing `api_key` or `project_id` in the .env file.")
    
    return api_key, watsonx_project_id

def load_css(file_name):
    """Load a CSS file for styling."""
    css_path = os.path.join("styles", file_name)  # Adjust to your folder
    try:
        with open(css_path) as file:
            return f'<style>{file.read()}</style>'
    except FileNotFoundError:
        logger.error(f"CSS file '{css_path}' not found.")
        raise
    except Exception as e:
        logger.error(f"Failed to load CSS file: {e}")
        raise

def create_collection_name(url):
    """Create a collection name from a URL."""
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
    """Initialize a ChromaDB client with custom cache settings."""
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
    """Clear a specific collection in ChromaDB."""
    try:
        collection = client.get_collection(collection_name)
        if collection:
            collection.delete()
            logger.info(f"Collection '{collection_name}' cleared successfully!")
    except ValueError:
        logger.warning(f"Collection '{collection_name}' does not exist, skipping.")
    except Exception as e:
        logger.error(f"Failed to clear collection '{collection_name}': {e}")
