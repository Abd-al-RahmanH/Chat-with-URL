import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
import requests
from bs4 import BeautifulSoup
import spacy
import chromadb
from utils import chromadb_client

# Load environment variables from the .env file
load_dotenv()

# Global constants
WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
API_KEY = os.getenv("api_key")
PROJECT_ID = os.getenv("project_id")

# Validate environment variables
if not API_KEY or not PROJECT_ID:
    raise ValueError("API Key or Project ID is missing from the .env file.")

# Model and decoding parameters
MODEL_PARAMS = {
    "model_type": "meta-llama/llama-3-70b-instruct",
    "max_tokens": 100,
    "min_tokens": 50,
    "decoding": DecodingMethods.GREEDY,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 1,
}

# Set up cache directory
CACHE_DIR = os.path.join(os.getcwd(), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR

# Load sentence-transformers model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(MODEL_NAME, cache_folder=CACHE_DIR)

# Embedding function
class MiniLML6V2EmbeddingFunction(EmbeddingFunction):
    MODEL = embedding_model

    def __call__(self, texts):
        return MiniLML6V2EmbeddingFunction.MODEL.encode(texts).tolist()


def get_model(params):
    """Retrieve the IBM WatsonX LLM model with specified parameters."""
    generate_params = {
        GenParams.MAX_NEW_TOKENS: params["max_tokens"],
        GenParams.MIN_NEW_TOKENS: params["min_tokens"],
        GenParams.DECODING_METHOD: params["decoding"],
        GenParams.TEMPERATURE: params["temperature"],
        GenParams.TOP_K: params["top_k"],
        GenParams.TOP_P: params["top_p"],
    }

    return Model(
        model_id=params["model_type"],
        params=generate_params,
        credentials={"apikey": API_KEY, "url": WATSONX_URL},
        project_id=PROJECT_ID,
    )


def extract_text(url):
    """Scrape text from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        cleaned_text = " ".join(paragraphs).replace("\xa0", " ")
        return cleaned_text
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from {url}: {e}")


def split_text_into_sentences(text):
    """Split text into sentences using SpaCy."""
    nlp = spacy.load("en_core_web_md")
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def create_embedding(url, collection_name, client):
    """Create embeddings for the text scraped from a URL."""
    text = extract_text(url)
    sentences = split_text_into_sentences(text)
    collection = client.get_or_create_collection(collection_name)
    collection.upsert(
        documents=sentences,
        metadatas=[{"source": str(i)} for i in range(len(sentences))],
        ids=[str(i) for i in range(len(sentences))],
    )
    return collection


def create_prompt(url, question, collection_name, client):
    """Generate a prompt using the embedded collection."""
    try:
        collection = create_embedding(url, collection_name, client)
        relevant_chunks = collection.query(query_texts=[question], n_results=5)
        context = "\n\n\n".join(relevant_chunks["documents"][0])
        prompt = (
            "<|begin_of_text|>\n"
            "<|start_header_id|>system<|end_header_id|>\n"
            "You are a helpful AI assistant.\n"
            "<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"### Context:\n{context}\n\n"
            f"### Instruction:\n"
            f"Please answer the following question based on the above context. Your answer should be concise and directly address the question. "
            f"If the question is unanswerable based on the given context, respond with 'unanswerable'.\n\n"
            f"### Question:\n{question}\n"
            "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
        return prompt
    except Exception as e:
        raise RuntimeError(f"Error creating prompt: {e}")


def answer_questions_from_web(url, question, collection_name, client):
    """Answer questions by querying WatsonX with relevant context."""
    model = get_model(MODEL_PARAMS)
    prompt = create_prompt(url, question, collection_name, client)
    generated_response = model.generate(prompt=prompt)
    return generated_response["results"][0]["generated_text"].strip()


def main():
    """Main function to run the RAG process."""
    client = chromadb_client()
    url = "https://huggingface.co/learn/nlp-course/chapter1/2?fw=pt"
    question = "What is NLP?"
    collection_name = "test_web_RAG"
    response = answer_questions_from_web(url, question, collection_name, client)
    print("Generated Response:")
    print(response)


if __name__ == "__main__":
    main()
